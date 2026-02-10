"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import argparse
import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *
from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from operator_data import eon_pkl_to_grid, load_pickle

# ------------------------------------------------------------
# Visualization helpers (PNG/PDF/SVG)
# ------------------------------------------------------------
import os
from viz_utils import (
    LearningCurve,
    plot_error_histogram,
    plot_learning_curve,
    plot_1d_prediction,
    rel_l2,
)

torch.manual_seed(0)
np.random.seed(0)


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, in_channels=1, out_channels=1):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(in_channels + 1, self.width) # input: (field channels, x)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.q = MLP(self.width, out_channels, self.width*2)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fourier Neural Operator 1D")
    add_data_mode_args(
        parser,
        default_data_mode="single_split",
        default_data_file="data/burgers_data_R10.mat",
        default_train_file=None,
        default_test_file=None,
    )
    parser.add_argument("--ntrain", type=int, default=1000, help="Number of training samples.")
    parser.add_argument("--ntest", type=int, default=100, help="Number of test samples.")
    parser.add_argument("--sub", type=int, default=2**3, help="Subsampling rate.")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs.")
    parser.add_argument("--modes", type=int, default=16, help="Number of Fourier modes.")
    parser.add_argument("--width", type=int, default=64, help="Model width.")
    parser.add_argument(
        "--dataset-format",
        choices=("mat", "eon_pkl"),
        default="mat",
        help="Input dataset format.",
    )
    parser.add_argument(
        "--eon-data-file",
        default=None,
        help="EON pkl file for single_split mode (fallback: --data-file).",
    )
    parser.add_argument(
        "--eon-train-file",
        default=None,
        help="EON pkl train file for separate_files mode (fallback: --train-file).",
    )
    parser.add_argument(
        "--eon-test-file",
        default=None,
        help="EON pkl test file for separate_files mode (fallback: --test-file).",
    )
    parser.add_argument(
        "--eon-meta-file",
        default=None,
        help="Meta pkl file for EON dataset conversion to grid.",
    )
    add_split_args(parser, default_train_split=0.8, default_seed=0)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validate_data_mode_args(args, parser)
    if args.dataset_format == "eon_pkl" and not args.eon_meta_file:
        parser.error("--eon-meta-file is required when --dataset-format=eon_pkl")


################################################################
#  configurations
################################################################
parser = _build_parser()
args = parser.parse_args()
_validate_args(args, parser)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

ntrain = args.ntrain
ntest = args.ntest

sub = args.sub #subsampling rate
s = 2**13 // sub

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
iterations = epochs*(ntrain//batch_size)

modes = args.modes
width = args.width

################################################################
# read data
################################################################

if args.dataset_format == "mat":
    # Data is of the shape (number of samples, grid size)
    if args.data_mode == "single_split":
        dataloader = MatReader(args.data_file)
        x_data = dataloader.read_field('a')[:,::sub]
        y_data = dataloader.read_field('u')[:,::sub]
        total = x_data.shape[0]
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

        x_train = x_data[train_idx]
        y_train = y_data[train_idx]
        x_test = x_data[test_idx]
        y_test = y_data[test_idx]
    else:
        train_reader = MatReader(args.train_file)
        test_reader = MatReader(args.test_file)

        x_train = train_reader.read_field('a')[:ntrain,::sub]
        y_train = train_reader.read_field('u')[:ntrain,::sub]
        x_test = test_reader.read_field('a')[-ntest:,::sub]
        y_test = test_reader.read_field('u')[-ntest:,::sub]
else:
    meta = load_pickle(args.eon_meta_file)
    if args.data_mode == "single_split":
        eon_data_file = args.eon_data_file or args.data_file
        x_data, y_data, _ = eon_pkl_to_grid(load_pickle(eon_data_file), meta)
        total = x_data.shape[0]
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
        x_train = x_data[train_idx]
        y_train = y_data[train_idx]
        x_test = x_data[test_idx]
        y_test = y_data[test_idx]
    else:
        eon_train_file = args.eon_train_file or args.train_file
        eon_test_file = args.eon_test_file or args.test_file
        x_train, y_train, _ = eon_pkl_to_grid(load_pickle(eon_train_file), meta)
        x_test, y_test, _ = eon_pkl_to_grid(load_pickle(eon_test_file), meta)
        x_train = x_train[:ntrain]
        y_train = y_train[:ntrain]
        x_test = x_test[:ntest]
        y_test = y_test[:ntest]

if x_train.ndim == 2:
    x_train = x_train.unsqueeze(-1)
if x_test.ndim == 2:
    x_test = x_test.unsqueeze(-1)
if y_train.ndim == 3 and y_train.shape[-1] == 1:
    y_train = y_train.squeeze(-1)
if y_test.ndim == 3 and y_test.shape[-1] == 1:
    y_test = y_test.squeeze(-1)

s = x_train.shape[1]
in_channels = x_train.shape[-1]

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# model
model = FNO1d(modes, width, in_channels=in_channels, out_channels=1).cuda()
print(count_params(model))

################################################################
# training and evaluation
################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)

# --- Visualization bookkeeping
viz_dir = os.path.join("visualizations", "fourier_1d")
os.makedirs(viz_dir, exist_ok=True)
hist_epochs: list[int] = []
hist_train_mse: list[float] = []
hist_train_rel_l2: list[float] = []
hist_test_rel_l2: list[float] = []
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)

        bsz = x.shape[0]
        mse = F.mse_loss(out.reshape(bsz, -1), y.reshape(bsz, -1), reduction='mean')
        l2 = myloss(out.reshape(bsz, -1), y.reshape(bsz, -1))
        l2.backward() # use the l2 relative loss

        optimizer.step()
        scheduler.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            bsz = x.shape[0]
            test_l2 += myloss(out.reshape(bsz, -1), y.reshape(bsz, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    # store history for plotting
    hist_epochs.append(ep)
    hist_train_mse.append(train_mse)
    hist_train_rel_l2.append(train_l2)
    hist_test_rel_l2.append(test_l2)

    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)

# ------------------------------------------------------------
# Visualize learning curves
# ------------------------------------------------------------
try:
    plot_learning_curve(
        LearningCurve(
            epochs=hist_epochs,
            train=hist_train_rel_l2,
            test=hist_test_rel_l2,
            train_label="train (relL2)",
            test_label="test (relL2)",
            metric_name="relative L2",
        ),
        out_path_no_ext=os.path.join(viz_dir, "learning_curve_relL2"),
        logy=True,
        title="fourier_1d: relative L2",
    )
    # MSE does not have a test curve here; plot train only (reuse API)
    plot_learning_curve(
        LearningCurve(
            epochs=hist_epochs,
            train=hist_train_mse,
            test=[np.nan] * len(hist_epochs),
            train_label="train (MSE)",
            test_label="",
            metric_name="MSE",
        ),
        out_path_no_ext=os.path.join(viz_dir, "learning_curve_mse"),
        logy=True,
        title="fourier_1d: train MSE",
    )
except Exception as e:
    print(f"[viz] failed to plot learning curves: {e}")

# torch.save(model, 'model/ns_fourier_burgers')
pred = torch.zeros(y_test.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        out = model(x).reshape(-1)
        pred[index] = out.detach().cpu()

        test_l2 += myloss(out.reshape(1, -1), y.reshape(1, -1)).item()
        print(index, test_l2)
        index = index + 1

# scipy.io.savemat('pred/burger_test.mat', mdict={'pred': pred.cpu().numpy()})

# ------------------------------------------------------------
# Visualize predictions on a few test samples
# ------------------------------------------------------------
try:
    # Per-sample relative L2 histogram
    per_sample_err = [rel_l2(pred[i], y_test[i]) for i in range(pred.shape[0])]
    plot_error_histogram(per_sample_err, os.path.join(viz_dir, "test_relL2_hist"))

    # Representative samples
    sample_ids = [0, min(1, ntest - 1), min(2, ntest - 1)]
    x_grid = np.linspace(0.0, 1.0, s)
    for i in sample_ids:
        plot_1d_prediction(
            x=x_grid,
            gt=y_test[i],
            pred=pred[i],
            input_u0=x_test[i].reshape(-1),
            out_path_no_ext=os.path.join(viz_dir, f"sample_{i:03d}"),
            title_prefix=f"sample {i}: ",
        )
except Exception as e:
    print(f"[viz] failed to plot sample predictions: {e}")
