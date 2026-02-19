"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which uses a recurrent structure to propagates in time.
"""

import argparse
import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer
from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args

# ------------------------------------------------------------
# Visualization helpers (PNG/PDF/SVG)
# ------------------------------------------------------------
import os
from viz_utils import (
    LearningCurve,
    plot_2d_time_slices,
    plot_error_histogram,
    plot_learning_curve,
    plot_rel_l2_over_time,
    rel_l2,
)

torch.manual_seed(0)
np.random.seed(0)

################################################################
# fourier layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels=12):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(in_channels, self.width) # input channel is T_in + 2 (x,y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, 1, self.width * 4) # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fourier Neural Operator 2D Time")
    add_data_mode_args(
        parser,
        default_data_mode="separate_files",
        default_data_file="data/ns_data_V100_N1000_T50_1.mat",
        default_train_file="data/ns_data_V100_N1000_T50_1.mat",
        default_test_file="data/ns_data_V100_N1000_T50_2.mat",
    )
    parser.add_argument("--ntrain", type=int, default=1000, help="Number of training samples.")
    parser.add_argument("--ntest", type=int, default=200, help="Number of test samples.")
    parser.add_argument("--modes", type=int, default=12, help="Number of Fourier modes.")
    parser.add_argument("--width", type=int, default=20, help="Model width.")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs.")
    parser.add_argument("--sub", type=int, default=1, help="Downsampling rate.")
    parser.add_argument("--S", type=int, default=64, help="Spatial grid size.")
    parser.add_argument("--T-in", type=int, default=10, help="Number of input timesteps.")
    parser.add_argument("--T", type=int, default=40, help="Number of prediction timesteps.")
    parser.add_argument("--step", type=int, default=1, help="Autoregressive step size.")
    add_split_args(parser, default_train_split=0.8, default_seed=0)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validate_data_mode_args(args, parser)


################################################################
# configs
################################################################
parser = _build_parser()
args = parser.parse_args()
_validate_args(args, parser)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

ntrain = args.ntrain
ntest = args.ntest

modes = args.modes
width = args.width

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
iterations = epochs*(ntrain//batch_size)

path = 'ns_fourier_2d_time_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path

# --- Visualization bookkeeping
viz_dir = path_image
os.makedirs(viz_dir, exist_ok=True)
hist_epochs: list[int] = []
hist_train_step: list[float] = []
hist_train_full: list[float] = []
hist_test_step: list[float] = []
hist_test_full: list[float] = []

sub = args.sub
S = args.S
T_in = args.T_in
T = args.T # T=40 for V1e-3; T=20 for V1e-4; T=10 for V1e-5;
step = args.step

################################################################
# load data
################################################################

if args.data_mode == "single_split":
    reader = MatReader(args.data_file)
    full_u = reader.read_field('u')
    total = full_u.shape[0]
    indices = np.arange(total)
    if args.shuffle:
        np.random.shuffle(indices)
    split_idx = int(total * args.train_split)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    if ntrain > len(train_idx) or ntest > len(test_idx):
        raise ValueError(
            f"Not enough samples for ntrain={ntrain}, ntest={ntest} with train split "
            f"{args.train_split} (total={total})."
        )

    train_idx = train_idx[:ntrain]
    test_idx = test_idx[:ntest]

    train_a = full_u[train_idx,::sub,::sub,:T_in]
    train_u = full_u[train_idx,::sub,::sub,T_in:T+T_in]
    test_a = full_u[test_idx,::sub,::sub,:T_in]
    test_u = full_u[test_idx,::sub,::sub,T_in:T+T_in]
else:
    reader = MatReader(args.train_file)
    train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
    train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]

    reader = MatReader(args.test_file)
    test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]
    test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

print(train_u.shape)
print(test_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain,S,S,T_in)
test_a = test_a.reshape(ntest,S,S,T_in)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################
model = FNO2d(modes, modes, width, in_channels=T_in + 2).to(device)
print(count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            bs = xx.shape[0]
            loss += myloss(im.reshape(bs, -1), y.reshape(bs, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(bs, -1), yy.reshape(bs, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                bs = xx.shape[0]
                loss += myloss(im.reshape(bs, -1), y.reshape(bs, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(bs, -1), yy.reshape(bs, -1)).item()

    t2 = default_timer()
    print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
          test_l2_full / ntest)

    # store history for plotting
    hist_epochs.append(ep)
    hist_train_step.append(train_l2_step / ntrain / (T / step))
    hist_train_full.append(train_l2_full / ntrain)
    hist_test_step.append(test_l2_step / ntest / (T / step))
    hist_test_full.append(test_l2_full / ntest)
# torch.save(model, path_model)

# ------------------------------------------------------------
# Visualize learning curves and a few qualitative rollouts
# ------------------------------------------------------------
try:
    plot_learning_curve(
        LearningCurve(
            epochs=hist_epochs,
            train=hist_train_step,
            test=hist_test_step,
            train_label="train (step relL2)",
            test_label="test (step relL2)",
            metric_name="relative L2",
        ),
        out_path_no_ext=os.path.join(viz_dir, "learning_curve_step_relL2"),
        logy=True,
        title="fourier_2d_time: stepwise relative L2",
    )
    plot_learning_curve(
        LearningCurve(
            epochs=hist_epochs,
            train=hist_train_full,
            test=hist_test_full,
            train_label="train (full relL2)",
            test_label="test (full relL2)",
            metric_name="relative L2",
        ),
        out_path_no_ext=os.path.join(viz_dir, "learning_curve_full_relL2"),
        logy=True,
        title="fourier_2d_time: full-trajectory relative L2",
    )

    def _rollout_autoregressive(xx0: torch.Tensor) -> torch.Tensor:
        """Roll out the model autoregressively for T steps (batch, S, S, T)."""
        xx = xx0.to(device)
        preds = []
        for _t in range(0, T, step):
            im = model(xx)
            preds.append(im)
            xx = torch.cat((xx[..., step:], im), dim=-1)
        return torch.cat(preds, dim=-1)

    model.eval()
    # Qualitative plots for a few test samples
    sample_ids = [0, min(1, ntest - 1), min(2, ntest - 1)]
    t_indices = [0, T // 2, T - 1]
    with torch.no_grad():
        for i in sample_ids:
            gt_i = test_u[i].cpu()
            pred_i = _rollout_autoregressive(test_a[i : i + 1]).squeeze(0).cpu()
            e_full = rel_l2(pred_i, gt_i)
            plot_2d_time_slices(
                gt=gt_i,
                pred=pred_i,
                t_indices=t_indices,
                out_path_no_ext=os.path.join(viz_dir, f"sample_{i:03d}_slices"),
                suptitle=f"sample {i}  full relL2={e_full:.3g}",
            )
            plot_rel_l2_over_time(
                gt=gt_i,
                pred=pred_i,
                out_path_no_ext=os.path.join(viz_dir, f"sample_{i:03d}_relL2_over_time"),
            )

    # Optional: histogram of full-trajectory errors on a subset (to keep runtime reasonable)
    n_hist = min(ntest, 50)
    per_sample_full = []
    with torch.no_grad():
        for i in range(n_hist):
            pred_i = _rollout_autoregressive(test_a[i : i + 1]).squeeze(0).cpu()
            per_sample_full.append(rel_l2(pred_i, test_u[i].cpu()))
    plot_error_histogram(
        per_sample_full,
        os.path.join(viz_dir, f"test_full_relL2_hist_first{n_hist}"),
        title=f"full relL2 histogram (first {n_hist} test samples)",
    )
except Exception as e:
    print(f"[viz] failed: {e}")
