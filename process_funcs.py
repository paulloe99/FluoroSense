import pandas as pd
import numpy as np
from scipy.special import erfc


def _require_lmfit():
    try:
        from lmfit import Model
        from lmfit.models import GaussianModel, LognormalModel
    except ImportError as exc:
        raise ImportError(
            "lmfit is required for model fitting. Install it with `pip install lmfit` "
            "or add `lmfit>=1.3.0` to your environment."
        ) from exc

    return Model, GaussianModel, LognormalModel


def _extract_spectrum(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if df.empty:
        raise ValueError("Spectrum dataframe is empty.")

    if df.shape[1] == 0:
        raise ValueError("Spectrum dataframe must contain at least one intensity column.")

    wavelength = df.index.to_numpy(dtype=float)
    intensity = df.iloc[:, 0].to_numpy(dtype=float)

    if wavelength.ndim != 1 or intensity.ndim != 1:
        raise ValueError("Spectrum data must be one-dimensional.")

    if len(wavelength) < 5:
        raise ValueError("At least 5 data points are required for derivative analysis.")

    if not np.all(np.diff(wavelength) > 0):
        raise ValueError("Spectrum index must be strictly increasing wavelength values.")

    return wavelength, intensity


def _nearest_index(x: np.ndarray, value: float) -> int:
    return int(np.abs(x - value).argmin())


def _local_minimum(x: np.ndarray, y: np.ndarray, center: float, half_width: float) -> tuple[float, float]:
    mask = (x >= center - half_width) & (x <= center + half_width)
    if not mask.any():
        idx = _nearest_index(x, center)
        return float(x[idx]), float(y[idx])

    local_x = x[mask]
    local_y = y[mask]
    idx = int(np.argmin(local_y))
    return float(local_x[idx]), float(local_y[idx])


def simple_fun(df: pd.DataFrame) -> float:
    return float(np.mean(df, axis=0).iloc[0])


def lambda_max_fun(df: pd.DataFrame) -> float:
    return float(df.iloc[:, 0].idxmax())


def aew_fun(df: pd.DataFrame) -> float:
    wavelength, intensity = _extract_spectrum(df)

    total_intensity = intensity.sum()
    return np.nan if total_intensity == 0 else float((wavelength * intensity).sum() / total_intensity)


def wavelength_ratio_fun(df: pd.DataFrame, wl_1, wl_2) -> float:
    wavelength, intensity = _extract_spectrum(df)
    idx_1 = _nearest_index(wavelength, float(wl_1))
    idx_2 = _nearest_index(wavelength, float(wl_2))

    return np.nan if intensity[idx_2] == 0 else float(intensity[idx_1] / intensity[idx_2])


def first_derivative_fun(df: pd.DataFrame) -> pd.DataFrame:
    wavelength, intensity = _extract_spectrum(df)
    first_derivative = np.gradient(intensity, wavelength)

    return pd.DataFrame(
        {
            "intensity": intensity,
            "first_derivative": first_derivative,
        },
        index=wavelength,
    )


def second_derivative_fun(
        df: pd.DataFrame,
        normalize_area: bool = False,
    ) -> pd.DataFrame:

    wavelength, intensity = _extract_spectrum(df)
    first_derivative = np.gradient(intensity, wavelength)
    second_derivative = np.gradient(first_derivative, wavelength)

    if normalize_area:
        area = np.trapz(np.abs(second_derivative), wavelength)
        if area > 0:
            second_derivative = second_derivative / area

    return pd.DataFrame(
        {
            "intensity": intensity,
            "first_derivative": first_derivative,
            "second_derivative": second_derivative,
        },
        index=wavelength,
    )


def derivative_analysis(
        df: pd.DataFrame,
        normalize_area: bool = True,
        class_centers: tuple[float, float, float] = (330.0, 340.0, 350.0),
        search_half_width: float = 5.0,
    ) -> dict[str, object]:
    """
    Analyze a fluorescence emission spectrum using the second-derivative method.

    The input dataframe must use wavelength as the index and intensity as the
    first column. The function smooths the spectrum, computes d2I/dlambda2,
    optionally area-normalizes the derivative, locates the local minima
    associated with the three characteristic tryptophan classes, and returns
    the H350/H330 derivative ratio.
    """

    wavelength, _ = _extract_spectrum(df)
    derivative_df = second_derivative_fun(
        df,
        normalize_area=normalize_area,
    )

    second_derivative = derivative_df["second_derivative"].to_numpy(dtype=float)


    class_results: dict[str, dict[str, float]] = {}

    for center in class_centers:
        peak_wavelength, derivative_minimum = _local_minimum(
            wavelength,
            second_derivative,
            center=center,
            half_width=search_half_width,
        )
        class_results[str(center)] = {
            "target_wavelength_nm": float(center),
            "peak_wavelength_nm": peak_wavelength,
            "derivative_minimum": derivative_minimum,
        }

    h330 = abs(class_results[str(class_centers[0])]["derivative_minimum"])
    h350 = abs(class_results[str(class_centers[2])]["derivative_minimum"])
    ratio = np.nan if h330 == 0 else float(h350 / h330)


    return {
        "class_peaks": pd.DataFrame(class_results).T,
        "h350_h330_ratio": ratio,
    }


def gaussian_fun(df: pd.DataFrame, visual_test: bool = False, verbose:bool = False):
    _, GaussianModel, _ = _require_lmfit()
    wavelength, intensity = _extract_spectrum(df)

    model = GaussianModel()
    params = model.guess(intensity, x=wavelength)
    result = model.fit(intensity, params, x=wavelength)

    if verbose:
        print("Fitted Gaussian distribution with following parameters:")
        print(result.best_values)
        print(f'Chi-square = {result.chisqr:.4f}, Reduced Chi-square = {result.redchi:.4f}')

    if visual_test:
        plt = result.plot()
        plt.show()
        input("Press Enter to continue, if residuals are normally distributed.")

    return result

def expgaussian_fun(df: pd.DataFrame, visual_test: bool = False, verbose:bool = False):
    Model, _, _ = _require_lmfit()

    def expgaussian(x, amplitude=1, center=0, sigma=1.0, gamma=1.0):
        dx = center - x
        return amplitude * np.exp(gamma * dx) * erfc(dx / (np.sqrt(2) * sigma))
    wavelength, intensity = _extract_spectrum(df)

    model =  Model(expgaussian)
    params = model.make_params(sigma=20, gamma=0.1, amplitude=400, center=350)
    result = model.fit(intensity, params, x=wavelength)

    if verbose:
        print("Fitted Gaussian distribution with following parameters:")
        print(result.best_values)
        print(f'Chi-square = {result.chisqr:.4f}, Reduced Chi-square = {result.redchi:.4f}')

    if visual_test:
        plt = result.plot()
        plt.show()
        input("Press Enter to continue, if residuals are normally distributed.")

    return result

def modelfit_fun(df: pd.DataFrame, visual_test: bool = False, verbose:bool = False):
    _, _, LognormalModel = _require_lmfit()
    wavelength, intensity = _extract_spectrum(df)

    model = LognormalModel()
    params = model.guess(intensity, x=wavelength)
    result = model.fit(intensity, params, x=wavelength)

    if verbose:
        print("Fitted Gaussian distribution with following parameters:")
        print(result.best_values)
        print(f'Chi-square = {result.chisqr:.4f}, Reduced Chi-square = {result.redchi:.4f}')

    if visual_test:
        plt = result.plot()
        plt.show()
        input("Press Enter to continue, if residuals are normally distributed.")

    return result
