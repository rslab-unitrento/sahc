import logging
import pathlib
import typing

import numpy as np
import pandas as pd
import torch
from torch import nn

T = typing.TypeVar("T", bound=np.generic, covariant=True)

Scalar = np.ndarray[tuple[()], np.dtype[T]]
Vector = np.ndarray[tuple[int], np.dtype[T]]
Matrix = np.ndarray[tuple[int, int], np.dtype[T]]
Tensor = np.ndarray[tuple[int, ...], np.dtype[T]]


class ConfMatrix(nn.Module):
    """
    Confusion Matrix class that handles batches of torch Tensors and provide
    several metrics.
    """

    def __init__(self, num_classes: int):
        """Create a Confusion Matrix object initialized with a specified number
        of classes.

        Parameters
        ----------
        num_classes : int
            The number of classes
        use_gpu : bool
            Whether to store the confusion matrix in the GPU
        """
        super(ConfMatrix, self).__init__()
        self.num_classes: int = num_classes
        self.state: torch.Tensor
        self.register_buffer(
            "state",
            torch.zeros(
                (self.num_classes, self.num_classes),
                dtype=torch.int64,
            ),
        )

    def reset(self) -> None:
        self.state.fill_(0)

    def calc(self, gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Calculates and returns the CM without saving it to state

        Parameters
        ----------
        gt : torch.Tensor
            Ground truth array
        pred : torch.Tensor
            Prediction array

        Returns
        -------
        torch.Tensor
            Numpy confusion matrix.
        """
        return torch.bincount(
            gt.long() * self.num_classes + pred, minlength=self.num_classes**2
        ).view(self.num_classes, self.num_classes)

    def get_existing_classes(self) -> torch.Tensor:
        """Returns the number of actual classes in the data, i.e., the number
        of classes with at least 1 reference/ground truth sample.

        Returns
        -------
        torch.Tensor
            The number of actual classes
        """
        return self.state.sum(dim=1).gt(0).sum()

    def add(
        self,
        gt: torch.Tensor,
        pred: torch.Tensor,
        ignore_index: int = 255,
    ):
        """Adds data to the confusion matrix.

        Parameters
        ----------
        gt : torch.Tensor
            Ground truth data
        pred : torch.Tensor
            Prediction data
        ignore_index : int, optional
            Nodata value, by default ``255``

        Raises
        ------
        ValueError
            `gt` and `pred` must have the same shape
        """
        logger = logging.getLogger("metrics.ConfMatrix.add")
        if gt.squeeze().size() != pred.squeeze().size():
            logger.error(
                f"`gt` {gt.squeeze().size()} and `pred` {pred.squeeze().size()} must have the same shape"
            )
            raise ValueError(
                f"`gt` {gt.squeeze().size()} and `pred` {pred.squeeze().size()} must have the same shape"
            )

        gt = gt.flatten()
        pred = pred.flatten()
        mask = gt.not_equal(ignore_index)
        pred = pred[mask]
        gt = gt[mask]

        if len(gt.size()) > 0:
            self.state += self.calc(gt, pred)

    def norm_on_lines(self) -> torch.Tensor:
        """Normalizes the confusion matrix along the lines.

        Returns
        -------
        torch.Tensor
            Normalized numpy confusion matrix
        """
        result = torch.zeros_like(self.state, dtype=torch.float32)
        divisor = self.state.sum(dim=1, keepdim=True)
        mask = divisor.gt(0)
        result[mask.squeeze(1)] = self.state[mask.squeeze(1)] / divisor[mask.squeeze(1)]
        return result

    def norm_on_cols(self) -> torch.Tensor:
        """Normalizes the confusion matrix along the cols.

        Returns
        -------
        torch.Tensor
            Normalized numpy confusion matrix
        """
        result = torch.zeros_like(self.state, dtype=torch.float32)
        divisor = self.state.sum(dim=0, keepdim=True)
        mask = divisor.gt(0)
        result[:, mask.squeeze(0)] = (
            self.state[:, mask.squeeze(0)] / divisor[:, mask.squeeze(0)]
        )
        return result

    def get_aa(self) -> torch.Tensor:
        """Return the Average Accuracy metric.

        Returns
        -------
        torch.Tensor
            Numpy scalar AA value.
        """
        confmatrix = self.norm_on_lines()
        return torch.diagonal(confmatrix).sum() / self.get_existing_classes()

    def get_oa(self) -> torch.Tensor:
        """Return the Overall Accuracy metric.

        Returns
        -------
        torch.Tensor
            Numpy scalar OA value.
        """
        return torch.diagonal(self.state).sum() / self.state.sum()

    def get_f1(self) -> torch.Tensor:
        """Return the Average Accuracy metric.

        Returns
        -------
        torch.Tensor
            Numpy scalar AA value.
        """

        confmatrixaa = self.norm_on_lines()
        confmatrixua = self.norm_on_cols()
        f1s = 2 * confmatrixaa * confmatrixua / (confmatrixaa + confmatrixua + 1e-9)
        return torch.diagonal(f1s).sum() / self.get_existing_classes()

    def get_IoU(self) -> torch.Tensor:
        """Return the Intersection over Union metric.

        Returns
        -------
        torch.Tensor
            Numpy IoU value for each class.
        """
        result = torch.zeros(
            self.num_classes, dtype=torch.float32, device=self.state.device
        )

        row_sum = self.state.sum(dim=0)
        col_sum = self.state.sum(dim=1)
        dividend = torch.diagonal(self.state)
        divisor = row_sum + col_sum - dividend

        # Divide only where doable
        mask = divisor.gt(0)
        result[mask] = dividend[mask] / divisor[mask]

        return result

    def get_mIoU(self) -> torch.Tensor:
        """Return the mean Intersection over Union metric.

        Returns
        -------
        torch.Tensor
            Numpy scalar mean IoU value.
        """
        return self.get_IoU().mean()


def save_classification_report(
    conf_mat: torch.Tensor | ConfMatrix,
    path: str | pathlib.Path,
    step: int,
    legend: list[str] | None = None,
    years: tuple[int, ...] | None = None,
) -> None:
    if isinstance(path, str):
        path = pathlib.Path(path)
    if isinstance(conf_mat, ConfMatrix):
        conf_mat = conf_mat.state

    if legend is None:
        legend = [str(i) for i in range(conf_mat.shape[-1])]

    diag = torch.diagonal(conf_mat)

    pas = torch.zeros(conf_mat.size(0), dtype=torch.float32, device=conf_mat.device)
    row_sums = conf_mat.sum(dim=1)
    # Divide only where doable
    mask = row_sums.gt(0)
    pas[mask] = diag[mask] / row_sums[mask]

    uas = torch.zeros(conf_mat.size(1), dtype=torch.float32, device=conf_mat.device)
    col_sums = conf_mat.sum(dim=0)
    # Divide only where doable
    mask = col_sums.gt(0)
    uas[mask] = diag[mask] / col_sums[mask]

    tot = row_sums.sum()

    f1s = 2 * pas * uas / (pas + uas + 1e-9)
    IoUs = diag / (row_sums + col_sums - diag)
    ref_prior = row_sums / tot
    map_prior = col_sums / tot
    data = torch.stack((uas, pas, f1s, IoUs, ref_prior, map_prior, row_sums), dim=1)
    df = pd.DataFrame(
        data=data.cpu().numpy(),
        index=legend,
        columns=[
            "UA",
            "PA",
            "F1",
            "IoU",
            "REF",
            "MAP",
            "REF_COUNT",
        ],
    )
    df.loc["Average"] = df.iloc[:, 0:4].mean()
    data = torch.stack(
        (
            uas * map_prior,
            pas * ref_prior,
            f1s * ref_prior,
            IoUs * ref_prior,
        ),
        dim=1,
    )
    df_wa = pd.DataFrame(
        data=data.cpu().numpy(),
        columns=[
            "UA",
            "PA",
            "F1",
            "IoU",
        ],
    )
    df.loc["Weighted Average"] = df_wa.sum()
    df.to_csv(path / f"classification_report_step_{step}.csv")


def classification_report(
    conf_mat: torch.Tensor | ConfMatrix,
    legend: list[str] | None = None,
) -> str:
    """Generate and return a classification report given a confusion matrix and
    an optional legend.

    Parameters
    ----------
    conf_mat : Union[torch.Tensor, ConfMatrix]
        Confusion Matrix
    legend : Optional[list[str]], optional
        Class legend as a list of string following the class order in the
        confusion matrix, by default None

    Returns
    -------
    str
        Classification report
    """
    logger = logging.getLogger("metrics.classification_report")

    if isinstance(conf_mat, ConfMatrix):
        conf_mat = conf_mat.state

    if legend is None:
        legend = [str(i) for i in range(conf_mat.shape[-1])]

    diag = torch.diagonal(conf_mat)

    pas = torch.zeros(conf_mat.size(0), dtype=torch.float32, device=conf_mat.device)
    row_sums = conf_mat.sum(dim=1)
    # Divide only where doable
    mask = row_sums.gt(0)
    pas[mask] = diag[mask] / row_sums[mask]

    uas = torch.zeros(conf_mat.size(1), dtype=torch.float32, device=conf_mat.device)
    col_sums = conf_mat.sum(dim=0)
    # Divide only where doable
    mask = col_sums.gt(0)
    uas[mask] = diag[mask] / col_sums[mask]

    tot = row_sums.sum()

    f1s = 2 * pas * uas / (pas + uas + 1e-9)
    IoUs = torch.nan_to_num((diag / (row_sums + col_sums - diag)), 0)
    ref_prior = row_sums / tot
    map_prior = col_sums / tot

    max_width = max(len(max(legend, key=len)), len("Weighted avg."))
    report = (
        f"{'Class':>{max_width}}  UA   PA   F1   IoU    REF     MAP     REF COUNT\n"
    )
    for i, label in enumerate(legend):
        report += (
            f"{label:>{max_width}} {uas[i]:.2f} {pas[i]:.2f} "
            f"{f1s[i]:.2f} {IoUs[i]:.2f} {ref_prior[i]:7.2%} "
            f"{map_prior[i]:7.2%} {row_sums[i].type(torch.int64):>12,d}\n"
        )
    report += f"{row_sums.sum().type(torch.int64):>{max_width + 49},d}\n"

    report += (
        f"\n{'Average':>{max_width}} {uas[map_prior > 0].mean():.2f} {pas[map_prior > 0].mean():.2f} "
        f"{f1s[map_prior > 0].mean():.2f} {IoUs[map_prior > 0].mean():.2f}\n"
    )
    report += (
        f"{'Weighted avg.':>{max_width}} {torch.sum(uas * map_prior):.2f} "
        f"{torch.sum(pas * ref_prior):.2f} {torch.sum(f1s * ref_prior):.2f} "
        f"{torch.sum(IoUs * ref_prior):.2f}"
    )
    logger.info(f"Classification report:\n{report}")
    return report
