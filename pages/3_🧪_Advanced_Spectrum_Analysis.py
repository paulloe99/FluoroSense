"""
FluoroSense - Advanced Spectrum Analysis
Second-derivative analysis and model fitting workflow.
"""
import inspect
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import process_funcs as pf
from styles import EMISSION_PALETTE, apply_dark_lab_theme, get_plotly_dark_template


st.set_page_config(
    layout="wide",
    page_title="Advanced Spectrum Analysis",
    page_icon="🧪",
    initial_sidebar_state="expanded",
)

apply_dark_lab_theme()

dark_template = get_plotly_dark_template()

config = {
    "displaylogo": False,
    "displayModeBar": True,
    "toImageButtonOptions": {
        "format": "svg",
        "filename": "advanced_spectrum_analysis",
        "height": 600,
        "width": 1200,
        "scale": 1,
    },
}


def _decode_uploaded_file(uploaded_file) -> str:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    return raw.decode("utf-8-sig", errors="replace")


def parse_spectrum_file(uploaded_file) -> tuple[dict, pd.DataFrame, dict]:
    text = _decode_uploaded_file(uploaded_file)
    header: dict[str, str] = {"TITLE": uploaded_file.name}
    extended_info: dict[str, str] = {}
    xydata: list[list[str]] = []

    if "XYDATA" not in text:
        df = pd.read_csv(StringIO(text), sep=None, engine="python")
        return header, df, extended_info

    mode = "header"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("XYDATA"):
            mode = "data"
            continue
        if line.startswith("##### Extended Information"):
            mode = "extended"
            continue

        if mode == "header":
            if "," in line:
                key, value = line.split(",", 1)
                header[key] = value.rstrip(",")
        elif mode == "data":
            if line.startswith("#####"):
                mode = "extended"
                continue
            xydata.append(line.split(","))
        elif mode == "extended" and "," in line:
            key, value = line.split(",", 1)
            extended_info[key.strip()] = value.strip()

    if not xydata or len(xydata) < 2:
        return header, pd.DataFrame(), extended_info

    df = pd.DataFrame(xydata[1:], columns=xydata[0])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return header, df.dropna(how="all"), extended_info


def to_spectrum_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("No tabular spectrum data found.")

    df = raw_df.copy()
    if "" in df.columns:
        df = df.rename(columns={"": "Wavelength [nm]"})

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) >= 2:
        wavelength_col = next(
            (col for col in numeric_cols if "wave" in str(col).lower() or "nm" in str(col).lower()),
            numeric_cols[0],
        )
        intensity_candidates = [col for col in numeric_cols if col != wavelength_col]
        intensity_col = next(
            (col for col in intensity_candidates if "int" in str(col).lower()),
            intensity_candidates[0],
        )
        spectrum = df[[wavelength_col, intensity_col]].dropna()
        spectrum.columns = ["wavelength", "intensity"]
        spectrum = spectrum.sort_values("wavelength")
        spectrum = spectrum.drop_duplicates("wavelength", keep="first")
        return spectrum.set_index("wavelength").rename_axis(None)

    if pd.api.types.is_numeric_dtype(df.index) and numeric_cols:
        spectrum = df[[numeric_cols[0]]].dropna()
        spectrum.columns = ["intensity"]
        return spectrum.sort_index().rename_axis(None)

    raise ValueError("Could not identify wavelength and intensity columns.")


def subtract_blank(sample_df: pd.DataFrame, blank_df: pd.DataFrame) -> pd.DataFrame:
    if blank_df.empty:
        return sample_df

    sample_wl = sample_df.index.to_numpy(dtype=float)
    blank_wl = blank_df.index.to_numpy(dtype=float)
    blank_intensity = blank_df.iloc[:, 0].to_numpy(dtype=float)
    corrected = sample_df.copy()
    corrected.iloc[:, 0] = corrected.iloc[:, 0] - np.interp(sample_wl, blank_wl, blank_intensity)
    return corrected


def spectrum_label(header: dict, fallback: str) -> str:
    return header.get("TITLE") or header.get("SAMPLE") or fallback


def calculate_summary(
    spectra: list[dict],
    class_centers: tuple[float, float, float],
    search_half_width: float,
    normalize_derivative: bool,
    ratio_wavelengths: tuple[float, float],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    class_tables = {}

    for item in spectra:
        label = item["label"]
        df = item["df"]
        derivative_result = pf.derivative_analysis(
            df,
            normalize_area=normalize_derivative,
            class_centers=class_centers,
            search_half_width=search_half_width,
        )
        class_tables[label] = derivative_result["class_peaks"].copy()
        rows.append(
            {
                "Spectrum": label,
                "Mean intensity": pf.simple_fun(df),
                "Lambda max [nm]": pf.lambda_max_fun(df),
                "AEW [nm]": pf.aew_fun(df),
                f"I{ratio_wavelengths[0]:.0f}/I{ratio_wavelengths[1]:.0f}": pf.wavelength_ratio_fun(
                    df,
                    ratio_wavelengths[0],
                    ratio_wavelengths[1],
                ),
                "H350/H330 derivative ratio": derivative_result["h350_h330_ratio"],
            }
        )

    return pd.DataFrame(rows), class_tables


def plot_spectra(spectra: list[dict]) -> go.Figure:
    fig = go.Figure()
    for idx, item in enumerate(spectra):
        df = item["df"]
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df.iloc[:, 0],
                mode="lines",
                name=item["label"],
                line=dict(color=EMISSION_PALETTE[idx % len(EMISSION_PALETTE)], width=2),
            )
        )

    fig.update_layout(
        template=dark_template,
        height=560,
        xaxis_title="Wavelength [nm]",
        yaxis_title="Intensity",
        legend_title="Spectrum",
    )
    return fig


def plot_derivatives(df: pd.DataFrame, label: str, normalize_area: bool) -> go.Figure:
    derivative_df = pf.second_derivative_fun(df, normalize_area=normalize_area)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=derivative_df.index,
            y=derivative_df["intensity"],
            mode="lines",
            name="Intensity",
            yaxis="y",
            line=dict(color=EMISSION_PALETTE[0], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=derivative_df.index,
            y=derivative_df["first_derivative"],
            mode="lines",
            name="First derivative",
            yaxis="y2",
            line=dict(color=EMISSION_PALETTE[1], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=derivative_df.index,
            y=derivative_df["second_derivative"],
            mode="lines",
            name="Second derivative",
            yaxis="y2",
            line=dict(color=EMISSION_PALETTE[2], width=2),
        )
    )
    fig.update_layout(
        template=dark_template,
        title=label,
        height=580,
        xaxis_title="Wavelength [nm]",
        yaxis=dict(title="Intensity"),
        yaxis2=dict(title="Derivative", overlaying="y", side="right", showgrid=False),
        legend_title="Trace",
    )
    return fig


def get_lmfit_model_classes() -> dict[str, type]:
    import lmfit.models as lmfit_models

    excluded_models = {
        "ExpressionModel",
        "Gaussian2dModel",
        "RectangleModel",
        "SplineModel",
    }
    model_classes = {}
    for name, obj in vars(lmfit_models).items():
        if not inspect.isclass(obj) or not name.endswith("Model") or name in excluded_models:
            continue
        try:
            signature = inspect.signature(obj)
            required_args = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
        except (TypeError, ValueError):
            continue

        if required_args and name != "PolynomialModel":
            continue

        try:
            model = obj(degree=2) if name == "PolynomialModel" else obj()
            if model.independent_vars != ["x"]:
                continue
        except Exception:
            continue
        model_classes[name] = obj

    return dict(sorted(model_classes.items()))


def create_lmfit_model(model_name: str, polynomial_degree: int = 2):
    model_classes = get_lmfit_model_classes()
    if model_name not in model_classes:
        raise ValueError(f"Unknown lmfit model: {model_name}")

    model_class = model_classes[model_name]
    if model_name == "PolynomialModel":
        return model_class(degree=polynomial_degree)
    return model_class()


def make_initial_params(model, df: pd.DataFrame):
    wavelength = df.index.to_numpy(dtype=float)
    intensity = df.iloc[:, 0].to_numpy(dtype=float)
    try:
        return model.guess(intensity, x=wavelength)
    except Exception:
        return model.make_params()


def params_to_editor_df(params) -> pd.DataFrame:
    rows = []
    for name, param in params.items():
        rows.append(
            {
                "Parameter": name,
                "Initial value": float(param.value) if param.value is not None else 0.0,
                "Min": "" if np.isneginf(param.min) else f"{param.min:.6g}",
                "Max": "" if np.isposinf(param.max) else f"{param.max:.6g}",
                "Vary": bool(param.vary),
            }
        )
    return pd.DataFrame(rows)


def parse_optional_float(value, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    parsed = float(value)
    return parsed


def apply_edited_params(params, edited_params_df: pd.DataFrame):
    for _, row in edited_params_df.iterrows():
        name = row["Parameter"]
        if name not in params:
            continue

        value = float(row["Initial value"])
        min_value = parse_optional_float(row["Min"], -np.inf)
        max_value = parse_optional_float(row["Max"], np.inf)
        if min_value >= max_value:
            raise ValueError(f"Parameter '{name}' has min >= max.")

        params[name].set(
            value=value,
            min=min_value,
            max=max_value,
            vary=bool(row["Vary"]),
        )
    return params


def fit_lmfit_model(
    df: pd.DataFrame,
    model_name: str,
    edited_params_df: pd.DataFrame,
    polynomial_degree: int = 2,
    fit_method: str = "leastsq",
    max_nfev: Optional[int] = None,
):
    wavelength = df.index.to_numpy(dtype=float)
    intensity = df.iloc[:, 0].to_numpy(dtype=float)
    model = create_lmfit_model(model_name, polynomial_degree=polynomial_degree)
    params = make_initial_params(model, df)
    params = apply_edited_params(params, edited_params_df)
    fit_kwargs = {"method": fit_method}
    if max_nfev:
        fit_kwargs["max_nfev"] = max_nfev
    return model.fit(intensity, params, x=wavelength, **fit_kwargs)


def plot_model_fit(df: pd.DataFrame, result, model_name: str) -> go.Figure:
    wavelength = df.index.to_numpy(dtype=float)
    intensity = df.iloc[:, 0].to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wavelength,
            y=intensity,
            mode="markers",
            name="Measured",
            marker=dict(color=EMISSION_PALETTE[0], size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=wavelength,
            y=result.best_fit,
            mode="lines",
            name=model_name,
            line=dict(color=EMISSION_PALETTE[3], width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=wavelength,
            y=result.residual,
            mode="lines",
            name="Residual",
            yaxis="y2",
            line=dict(color=EMISSION_PALETTE[4], width=1),
        )
    )
    fig.update_layout(
        template=dark_template,
        height=580,
        xaxis_title="Wavelength [nm]",
        yaxis=dict(title="Intensity"),
        yaxis2=dict(title="Residual", overlaying="y", side="right", showgrid=False),
        legend_title="Trace",
    )
    return fig


st.title("Advanced Spectrum Analysis")
st.markdown("Second-derivative classification, wavelength-ratio metrics, and optional model fitting for emission spectra.")

st.sidebar.header("Data Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload spectra",
    type=["csv", "txt"],
    accept_multiple_files=True,
    key="advanced_spectra",
)

blank_file = st.sidebar.file_uploader(
    "Upload blank spectrum",
    type=["csv", "txt"],
    accept_multiple_files=False,
    key="advanced_blank",
)
use_blank = st.sidebar.checkbox("Subtract blank", value=False)

st.sidebar.markdown("---")
st.sidebar.header("Derivative Settings")
center_330 = st.sidebar.number_input("Class 330 center [nm]", value=330.0, step=1.0)
center_340 = st.sidebar.number_input("Class 340 center [nm]", value=340.0, step=1.0)
center_350 = st.sidebar.number_input("Class 350 center [nm]", value=350.0, step=1.0)
search_half_width = st.sidebar.number_input("Search half-width [nm]", min_value=0.5, value=5.0, step=0.5)
normalize_derivative = st.sidebar.checkbox("Area-normalize second derivative", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Ratio Metric")
ratio_wl_1 = st.sidebar.number_input("Numerator wavelength [nm]", value=350.0, step=1.0)
ratio_wl_2 = st.sidebar.number_input("Denominator wavelength [nm]", value=330.0, step=1.0)

if not uploaded_files:
    st.info("Upload one or more emission spectra to start the advanced workflow.")
    st.stop()

blank_df = pd.DataFrame()
if use_blank and blank_file is not None:
    try:
        _, raw_blank_df, _ = parse_spectrum_file(blank_file)
        blank_df = to_spectrum_df(raw_blank_df)
        st.sidebar.success("Blank loaded")
    except Exception as exc:
        st.sidebar.error(f"Blank could not be loaded: {exc}")
elif use_blank:
    st.sidebar.warning("Upload a blank spectrum to apply subtraction.")

spectra = []
load_errors = []
for uploaded_file in uploaded_files:
    try:
        header, raw_df, extended_info = parse_spectrum_file(uploaded_file)
        spectrum_df = to_spectrum_df(raw_df)
        if use_blank and not blank_df.empty:
            spectrum_df = subtract_blank(spectrum_df, blank_df)
        spectra.append(
            {
                "file_name": uploaded_file.name,
                "label": spectrum_label(header, uploaded_file.name),
                "header": header,
                "extended_info": extended_info,
                "df": spectrum_df,
            }
        )
    except Exception as exc:
        load_errors.append(f"{uploaded_file.name}: {exc}")

for error in load_errors:
    st.error(error)

if not spectra:
    st.stop()

class_centers = (center_330, center_340, center_350)
ratio_wavelengths = (ratio_wl_1, ratio_wl_2)

try:
    summary_df, class_tables = calculate_summary(
        spectra,
        class_centers=class_centers,
        search_half_width=search_half_width,
        normalize_derivative=normalize_derivative,
        ratio_wavelengths=ratio_wavelengths,
    )
except Exception as exc:
    st.error(f"Analysis failed: {exc}")
    st.stop()

tab_overview, tab_derivatives, tab_model, tab_export = st.tabs(
    ["Overview", "Derivative Analysis", "Model Fit", "Export"]
)

with tab_overview:
    st.subheader("Spectra")
    st.plotly_chart(plot_spectra(spectra), width="stretch", config=config)
    st.subheader("Metrics")
    st.dataframe(summary_df, width="stretch", hide_index=True)

with tab_derivatives:
    selected_label = st.selectbox("Spectrum", [item["label"] for item in spectra], key="derivative_spectrum")
    selected_item = next(item for item in spectra if item["label"] == selected_label)
    st.plotly_chart(
        plot_derivatives(selected_item["df"], selected_label, normalize_derivative),
        width="stretch",
        config=config,
    )

    st.subheader("Derivative Class Minima")
    st.dataframe(class_tables[selected_label], width="stretch")

    derivative_download = pf.second_derivative_fun(
        selected_item["df"],
        normalize_area=normalize_derivative,
    ).reset_index(names="Wavelength [nm]")
    st.download_button(
        "Download derivative data",
        data=derivative_download.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_label}_derivatives.csv",
        mime="text/csv",
    )

with tab_model:
    selected_label = st.selectbox("Spectrum", [item["label"] for item in spectra], key="fit_spectrum")
    selected_item = next(item for item in spectra if item["label"] == selected_label)

    try:
        model_classes = get_lmfit_model_classes()
        preferred_models = [
            "GaussianModel",
            "ExponentialGaussianModel",
            "LognormalModel",
            "LorentzianModel",
            "VoigtModel",
            "PseudoVoigtModel",
            "SkewedGaussianModel",
            "StudentsTModel",
        ]
        model_options = [model for model in preferred_models if model in model_classes]
        model_options.extend([model for model in model_classes if model not in model_options])

        model_filter = st.text_input("Filter lmfit models", value="", placeholder="Type to filter model names")
        filtered_model_options = [
            model for model in model_options if model_filter.lower() in model.lower()
        ] or model_options
        default_model_index = filtered_model_options.index("GaussianModel") if "GaussianModel" in filtered_model_options else 0
        model_name = st.selectbox(
            "lmfit model",
            filtered_model_options,
            index=default_model_index,
            help="One-dimensional lmfit model classes are listed. Multi-dimensional and expression models are omitted.",
        )

        polynomial_degree = 2
        if model_name == "PolynomialModel":
            polynomial_degree = st.number_input("Polynomial degree", min_value=1, max_value=12, value=2, step=1)

        model = create_lmfit_model(model_name, polynomial_degree=polynomial_degree)
        initial_params = make_initial_params(model, selected_item["df"])
        st.subheader("Initial Fit Parameters")
        initial_params_df = params_to_editor_df(initial_params)
        edited_params_df = st.data_editor(
            initial_params_df,
            width="stretch",
            hide_index=True,
            disabled=["Parameter"],
            key=f"params_{selected_label}_{model_name}_{polynomial_degree}",
            column_config={
                "Initial value": st.column_config.NumberColumn(format="%.6g"),
                "Min": st.column_config.TextColumn(help="Leave empty for -infinity."),
                "Max": st.column_config.TextColumn(help="Leave empty for infinity."),
                "Vary": st.column_config.CheckboxColumn(),
            },
        )

        fit_col1, fit_col2 = st.columns(2)
        with fit_col1:
            fit_method = st.selectbox(
                "Fit method",
                ["leastsq", "least_squares", "nelder", "powell", "lbfgsb", "cg", "bfgs"],
                index=0,
            )
        with fit_col2:
            max_nfev = st.number_input(
                "Max function evaluations",
                min_value=0,
                value=0,
                step=100,
                help="Use 0 for lmfit's default.",
            )

        if st.button("Run fit", type="primary"):
            fit_result = fit_lmfit_model(
                selected_item["df"],
                model_name,
                edited_params_df,
                polynomial_degree=polynomial_degree,
                fit_method=fit_method,
                max_nfev=None if max_nfev == 0 else int(max_nfev),
            )
            st.session_state["advanced_fit_result"] = fit_result
            st.session_state["advanced_fit_model"] = model_name
            st.session_state["advanced_fit_spectrum"] = selected_label

        fit_result = st.session_state.get("advanced_fit_result")
        fit_model = st.session_state.get("advanced_fit_model")
        fit_spectrum = st.session_state.get("advanced_fit_spectrum")

        if fit_result is not None and fit_model == model_name and fit_spectrum == selected_label:
            st.plotly_chart(
                plot_model_fit(selected_item["df"], fit_result, model_name),
                width="stretch",
                config=config,
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Chi-square", f"{fit_result.chisqr:.4g}")
            col2.metric("Reduced chi-square", f"{fit_result.redchi:.4g}")
            col3.metric("R²", f"{getattr(fit_result, 'rsquared', np.nan):.4g}")

            params_df = pd.DataFrame(
                [
                    {
                        "Parameter": name,
                        "Value": param.value,
                        "Standard error": param.stderr,
                        "Min": param.min,
                        "Max": param.max,
                        "Vary": param.vary,
                    }
                    for name, param in fit_result.params.items()
                ]
            )
            st.subheader("Fit Parameters")
            st.dataframe(params_df, width="stretch", hide_index=True)

            fit_export = pd.DataFrame(
                {
                    "Wavelength [nm]": selected_item["df"].index,
                    "Intensity": selected_item["df"].iloc[:, 0].to_numpy(dtype=float),
                    "Fit": fit_result.best_fit,
                    "Residual": fit_result.residual,
                }
            )
            st.download_button(
                "Download fit data",
                data=fit_export.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_label}_{model_name.lower()}_fit.csv",
                mime="text/csv",
            )
    except ImportError as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.error(f"Model fitting failed: {exc}")

with tab_export:
    st.subheader("Summary CSV")
    st.download_button(
        "Download metrics summary",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="advanced_spectrum_metrics.csv",
        mime="text/csv",
    )

    peak_rows = []
    for label, peaks_df in class_tables.items():
        tmp = peaks_df.reset_index(names="Class center")
        tmp.insert(0, "Spectrum", label)
        peak_rows.append(tmp)

    class_peaks_export = pd.concat(peak_rows, ignore_index=True)
    st.subheader("Derivative Class Minima CSV")
    st.dataframe(class_peaks_export, width="stretch", hide_index=True)
    st.download_button(
        "Download derivative class minima",
        data=class_peaks_export.to_csv(index=False).encode("utf-8"),
        file_name="advanced_derivative_class_minima.csv",
        mime="text/csv",
    )
