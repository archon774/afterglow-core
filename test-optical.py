#!/usr/bin/env python
"""
test-optical.py

Batch-test the legacy Afterglow optical stack (WCS calibration, aperture
photometry, field calibration) on all FITS files in test-fits/.

Run as:
    /home/claude/venvs/afterglow/bin/python /home/claude/afterglow-core/test-optical.py

No CLI arguments. All settings are fixed constants defined below.

Zero-point note
---------------
The solved photometric zero point is produced by field calibration via
calc_solution() and stored in FieldCalResult.zero_point_corr / FITS PHOT_M0.
PhotSettings.zero_point is only an additive instrument offset applied to
instrumental magnitudes and is NOT the fitted calibration result.

Reduction note
--------------
reduction_attempted and reduction_ok are always False. This script exercises
WCS, photometry, and field calibration — not bias/dark/flat pipeline reduction.

Non-default settings
--------------------
No scientific settings (aperture sizes, catalog names, match tolerances, SNR
limits, rejection parameters, filter mappings, etc.) are invented or hard-coded.
Only infrastructure values are fixed: repository paths, TMPDIR, ANET index paths,
and CSV column names. All other settings come from legacy model/schema defaults.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard library — imported before any Afterglow code
# ─────────────────────────────────────────────────────────────────────────────

import csv
import logging
import os
import shutil
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Optional

import astropy.io.fits as pyfits
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure constants  (paths only — no scientific settings)
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT  = Path("/home/claude/afterglow-core")
INPUT_DIR  = REPO_ROOT / "test-fits"
OUTPUT_CSV = REPO_ROOT / "test-optical-results.csv"
TMPDIR     = REPO_ROOT / "test_subjects" / "tmp" / "test-optical"

# Astrometry.net index directories — external infrastructure, not science settings
ANET_INDEX_PATH = [
    "/home/claude/catalogs/astrometry/UCAC5",
    "/home/claude/catalogs/astrometry/2MASS_ANET/4200",
    "/home/claude/catalogs/astrometry/TYCHO2/indices",
]

FITS_GLOB = "*.fits"

# ─────────────────────────────────────────────────────────────────────────────
# Scientific settings — explicitly specified; used for source extraction,
# aperture photometry, and field calibration
# ─────────────────────────────────────────────────────────────────────────────

# Source extraction
SE_THRESHOLD        = 2.5
SE_BK_SIZE          = 1/64      # 0.015625 of the image dimension
SE_BK_FILTER_SIZE   = 3
SE_FWHM             = 0
SE_MIN_FWHM         = 0.8
SE_MAX_FWHM         = 50.0
SE_MIN_PIXELS       = 3
SE_MAX_ELLIPTICITY  = 3.0
SE_CLEAN            = 1.0
SE_DEBLEND_LEVELS   = 32
SE_DEBLEND_CONTRAST = 0.005
SE_SAT_LEVEL        = 63000.0
SE_DOWNSAMPLE       = 1
SE_CLIP_LO          = 0.0
SE_CLIP_HI          = 100.0

# Aperture photometry (circular apertures: b/b_out left None → treated as a/a_out)
PHOT_APERTURE        = 5.0   # semi-major axis, pixels
PHOT_ANNULUS_INNER   = 10.0  # inner annulus semi-major axis, pixels
PHOT_ANNULUS_OUTER   = 15.0  # outer annulus semi-major axis, pixels
PHOT_CENTROID_RADIUS = 5.0   # centroiding search radius, pixels
PHOT_GAIN            = 1.0   # detector gain, e⁻/ADU
PHOT_ZERO_POINT      = 20.0  # instrumental magnitude zero-point offset

# Field calibration
FCAL_SOURCE_MATCH_TOL = 5.0    # source position match tolerance, pixels
FCAL_SOURCE_INCL_PCT  = 100.0  # source inclusion percentage
FCAL_MIN_SNR          = 10.0   # minimum signal-to-noise ratio for reference stars

CSV_COLUMNS = [
    "input_file", "object_key", "observation_asset_id", "processing_run_id",
    "output_file_id", "output_object_key", "run_status", "failure_reason",
    "exception_type", "exception_message", "validation_ok",
    "reduction_attempted", "reduction_ok",
    "wcs_attempted", "wcs_solved",
    "photometry_attempted", "photometry_ok",
    "field_cal_attempted", "field_cal_ok",
    "output_created",
    "wcs_crval1", "wcs_crval2", "wcs_crpix1", "wcs_crpix2",
    "wcs_cd1_1", "wcs_cd1_2", "wcs_cd2_1", "wcs_cd2_2",
    "wcs_ctype1", "wcs_ctype2", "wcs_header_written",
    "rej_percent", "zero_point", "zero_point_error", "zero_point_slop",
    "limmag5", "catalog_name", "filter_name",
    "elapsed_seconds", "notes",
]

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup — before any Afterglow imports that might emit messages
# ─────────────────────────────────────────────────────────────────────────────

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    warnings.filterwarnings("ignore")

configure_logging()
logger = logging.getLogger("test_optical")

# ─────────────────────────────────────────────────────────────────────────────
# Afterglow environment configuration
#
# afterglow_core/__init__.py calls create_app() at MODULE LOAD TIME.
# We must write the config file and set env vars BEFORE the first import of
# afterglow_core so that create_app() uses our SQLite database and our paths.
#
# The SQLite URI formula in afterglow_core/database.py is:
#   f'{DB_BACKEND}://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_SCHEMA}'
# With DB_BACKEND='sqlite', empty user/host, and port=0 this becomes:
#   sqlite://@:0//absolute/path/to/db
# which SQLAlchemy's SQLite driver accepts (authority section is ignored).
# ─────────────────────────────────────────────────────────────────────────────

# Stable paths for the SQLite DB and data root — placed alongside TMPDIR so
# that reset_tmpdir() (which wipes TMPDIR only) does not destroy them.
_DB_DIR      = REPO_ROOT / "test_subjects" / "tmp"
_DB_PATH     = _DB_DIR / "afterglow_test.db"
_DATA_ROOT   = _DB_DIR / "afterglow_data"
_CONFIG_PATH = _DB_DIR / "afterglow_test_config.py"
_LOG_DIR     = _DB_DIR / "afterglow_logs"


def configure_afterglow_environment() -> None:
    """
    Write a minimal Afterglow config file and set environment variables so that
    afterglow_core.create_app() succeeds with a local SQLite database.
    Must be called before the first ``import afterglow_core``.
    """
    for d in (_DB_DIR, _DATA_ROOT, _LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    _CONFIG_PATH.write_text(
        f"# Auto-generated Afterglow batch-test config — do not edit manually\n"
        f"AUTH_ENABLED = False\n"
        f"APPLICATION_ROOT = ''\n"
        f"DB_BACKEND = 'sqlite'\n"
        f"DB_USER = ''\n"
        f"DB_PASS = ''\n"
        f"DB_HOST = ''\n"
        f"DB_PORT = 0\n"
        f"DB_SCHEMA = '{_DB_PATH}'\n"
        f"DATA_ROOT = '{_DB_DIR}'\n"
        f"DATA_FILE_ROOT = '{_DATA_ROOT}'\n"
        f"DATA_FILE_COMPRESSION = False\n"
        f"DEBUG = True\n"
        f"ANET_INDEX_PATH = {ANET_INDEX_PATH!r}\n"
        # vizier.u-strasbg.fr redirects POST→GET (301), dropping the query body.
        # vizier.cds.unistra.fr is the canonical current address of VizieR.
        f"VIZIER_SERVER = 'vizier.cds.unistra.fr'\n"
        # Disable caching in the test script to prevent stale cached empty
        # responses (e.g. from a previous misconfigured server) from masking
        # real catalog query results.
        f"VIZIER_CACHE = False\n",
        encoding="utf-8",
    )
    os.environ["AFTERGLOW_LOGGING_ROOT"] = str(_LOG_DIR)
    os.environ["AFTERGLOW_CORE_CONFIG"]  = str(_CONFIG_PATH)
    logger.info("Afterglow environment configured; SQLite DB → %s", _DB_PATH)


# Write config and set env vars BEFORE any afterglow_core import
configure_afterglow_environment()

# ─────────────────────────────────────────────────────────────────────────────
# Import Afterglow — create_app() runs here against our SQLite config
# ─────────────────────────────────────────────────────────────────────────────

logger.info("Importing afterglow_core (create_app() will run with SQLite config)…")
sys.path.insert(0, str(REPO_ROOT))

import afterglow_core                                                  # noqa: E402
from afterglow_core import app as afterglow_app                        # noqa: E402
from afterglow_core.database import db                                 # noqa: E402
from afterglow_core.resources.data_files import (                      # noqa: E402
    create_data_file, get_data_file_path, get_root,
)
from afterglow_core.resources.field_cals import query_field_cals       # noqa: E402
from afterglow_core.resources.catalogs import catalogs as known_catalogs  # noqa: E402
from afterglow_core.models import FieldCal, PhotSettings, JobState     # noqa: E402
from afterglow_core.resources.job_plugins.wcs_calibration_job import ( # noqa: E402
    WcsCalibrationJob, WcsCalibrationJobResult,
)
from afterglow_core.resources.job_plugins.photometry_job import (      # noqa: E402
    run_photometry_job,
)
from afterglow_core.resources.job_plugins.field_cal_job import (       # noqa: E402
    FieldCalJob, FieldCalJobResult,
)
from afterglow_core.resources.job_plugins.source_extraction_job import (  # noqa: E402
    SourceExtractionSettings, run_source_extraction_job,
)

logger.info("afterglow_core loaded successfully")

# ─────────────────────────────────────────────────────────────────────────────
# MockTask: provides the Celery Task interface required by Job.update()
#
# When we call job.run() directly (outside the job server), job.update() is
# called internally. It calls self._task.update_state() which we stub out.
# ─────────────────────────────────────────────────────────────────────────────

class MockTask:
    """No-op Celery task stub for direct job execution (no job server)."""
    def update_state(self, task_id=None, state=None, meta=None):
        pass  # no Celery broker needed


# ─────────────────────────────────────────────────────────────────────────────
# Startup validators
# ─────────────────────────────────────────────────────────────────────────────

def validate_anet_indexes(paths: list) -> None:
    existing = [p for p in paths if Path(p).is_dir()]
    missing  = [p for p in paths if not Path(p).is_dir()]
    for p in missing:
        logger.warning("ANET index path not found: %s", p)
    if not existing:
        raise SystemExit(f"ERROR: No ANET index paths exist. Checked: {paths}")
    logger.info("ANET index paths found: %s", existing)


def reset_tmpdir() -> None:
    """Clear and recreate the per-run working directory."""
    if TMPDIR.exists():
        shutil.rmtree(TMPDIR)
    TMPDIR.mkdir(parents=True)
    logger.info("Fresh TMPDIR: %s", TMPDIR)


def iter_fits_files() -> list:
    if not INPUT_DIR.exists():
        raise SystemExit(f"ERROR: INPUT_DIR does not exist: {INPUT_DIR}")
    files = (
        sorted(INPUT_DIR.glob(FITS_GLOB))
        + sorted(INPUT_DIR.glob("*.fit"))
        + sorted(INPUT_DIR.glob("*.fits.gz"))
    )
    seen, unique = set(), []
    for f in files:
        if f not in seen:
            seen.add(f); unique.append(f)
    if not unique:
        raise SystemExit(f"ERROR: No FITS files found in {INPUT_DIR}")
    logger.info("Found %d FITS file(s) in %s", len(unique), INPUT_DIR)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# Data file registration
# ─────────────────────────────────────────────────────────────────────────────

def register_legacy_data_file(input_path: Path) -> int:
    """
    Read a FITS file and register it with the legacy Afterglow data file system.
    Returns the integer file_id assigned by the database.

    The original file is never modified — data is read into memory and stored
    as a new data file in the Afterglow data root.
    """
    with pyfits.open(str(input_path), memmap=False) as hdul:
        data = np.array(hdul[0].data, dtype=np.float32)
        hdr  = hdul[0].header.copy()

    root = get_root(None)
    os.makedirs(root, exist_ok=True)

    db_file = create_data_file(
        user_id    = None,
        name       = input_path.name,
        root       = root,
        data       = data,
        hdr        = hdr,
        duplicates = 'append',
        session_id = None,
    )
    db.session.commit()
    return db_file.id


# ─────────────────────────────────────────────────────────────────────────────
# WCS calibration
# ─────────────────────────────────────────────────────────────────────────────

def run_legacy_wcs(file_id: int) -> tuple:
    """
    Run WCS calibration using the legacy WcsCalibrationJob plugin with all
    schema defaults:
      - ra_hours / dec_degs: None (plugin infers from FITS header)
      - radius: 180 deg, min_scale: 0.1, max_scale: 60 arcsec/px
      - sip_order: 0, crpix_center: True, max_sources: 100
      - source_extraction_settings: None (uses SourceExtractionSettings defaults)
      - inplace: True (default — modifies the data file in place)

    ANET_INDEX_PATH is an external infrastructure setting already configured
    in app.config by configure_afterglow_environment().

    Returns (solved_file_id_or_None, meta_dict).
    """
    job = WcsCalibrationJob()
    job.user_id    = None
    job.session_id = None
    job.id         = str(uuid.uuid4())
    job.state      = JobState()
    job.state.status = 'in_progress'
    job.result     = WcsCalibrationJobResult()
    job._task      = MockTask()
    job.file_ids   = [file_id]
    # settings field uses dump_default={} → WcsCalibrationSettings() with all defaults
    # source_extraction_settings uses dump_default=None → None (plugin uses its own defaults)
    # inplace uses dump_default=True → True

    job.run()

    meta = {
        'errors': job.result.errors,
        'warnings': job.result.warnings,
    }
    if job.result.errors or not job.result.file_ids:
        return None, meta
    return job.result.file_ids[0], meta


# ─────────────────────────────────────────────────────────────────────────────
# Standalone aperture photometry
#
# NOTE: The solved photometric zero point is NOT produced here.
#       PhotSettings.zero_point is only an additive offset applied to
#       instrumental magnitudes. The fitted zero point comes exclusively from
#       field calibration (calc_solution → FieldCalResult.zero_point_corr /
#       FITS PHOT_M0). This function exercises the legacy photometry path for
#       completeness but its zero_point output is not the calibration result.
# ─────────────────────────────────────────────────────────────────────────────

class _StubJob:
    """Minimal job-like object for calling run_*_job helper functions."""
    def __init__(self):
        self.user_id    = None
        self.session_id = None
        self.id         = str(uuid.uuid4())
        self.state      = type('S', (), {'status': 'in_progress', 'progress': 0})()
        self._task      = MockTask()
        self._errors    = []

    def add_error(self, e, meta=None):
        self._errors.append(str(e))

    def add_warning(self, msg):
        pass

    def update_progress(self, *args, **kwargs):
        pass

    def update(self):
        pass


def make_phot_settings() -> 'PhotSettings':
    """Return a PhotSettings instance with the specified values."""
    s = PhotSettings()
    s.a               = PHOT_APERTURE
    s.a_in            = PHOT_ANNULUS_INNER
    s.a_out           = PHOT_ANNULUS_OUTER
    s.centroid_radius = PHOT_CENTROID_RADIUS
    s.gain            = PHOT_GAIN
    s.zero_point      = PHOT_ZERO_POINT
    return s


def make_source_extraction_settings() -> 'SourceExtractionSettings':
    """Return a SourceExtractionSettings instance with the specified values."""
    s = SourceExtractionSettings()
    s.threshold        = SE_THRESHOLD
    s.bk_size          = SE_BK_SIZE
    s.bk_filter_size   = SE_BK_FILTER_SIZE
    s.fwhm             = SE_FWHM
    s.min_fwhm         = SE_MIN_FWHM
    s.max_fwhm         = SE_MAX_FWHM
    s.min_pixels       = SE_MIN_PIXELS
    s.max_ellipticity  = SE_MAX_ELLIPTICITY
    s.clean            = SE_CLEAN
    s.deblend_levels   = SE_DEBLEND_LEVELS
    s.deblend_contrast = SE_DEBLEND_CONTRAST
    s.sat_level        = SE_SAT_LEVEL
    s.downsample       = SE_DOWNSAMPLE
    s.clip_lo          = SE_CLIP_LO
    s.clip_hi          = SE_CLIP_HI
    return s


def run_legacy_aperture_photometry(file_id: int, sources: list) -> list:
    """
    Standalone aperture photometry using the specified PhotSettings.
    """
    settings = make_phot_settings()
    stub = _StubJob()
    return run_photometry_job(stub, settings, [file_id], sources)


# ─────────────────────────────────────────────────────────────────────────────
# Field calibration
#
# NOTE: Field calibration via calc_solution() is the AUTHORITATIVE source for
#       the fitted zero_point. The result is stored in:
#         - FieldCalResult.zero_point_corr  (primary)
#         - FITS header PHOT_M0             (secondary, written by the job)
#       Do not use PhotSettings.zero_point as the solved zero point — it is
#       only an additive instrument offset, not a fitted calibration value.
# ─────────────────────────────────────────────────────────────────────────────

def build_default_field_cal(user_id: Optional[int]) -> tuple:
    """
    Return a FieldCal to use for calibration.

    1. Check for stored presets; use the first if found.
    2. Otherwise discover catalog plugins from the registry and build a FieldCal
       with the specified FCAL_* parameters.
    """
    # 1. Try stored presets (the legacy normal workflow)
    try:
        stored = query_field_cals(user_id)
        if stored:
            fc = stored[0]
            logger.info("Using stored field cal preset: id=%s name=%s", fc.id, fc.name)
            return fc, ""
    except Exception as exc:
        logger.warning("Could not query stored field cals: %s", exc)

    # 2. No stored presets — discover available catalog plugins
    # Exclude SDSS: its astroquery plugin passes a tuple to Angle() which
    # raises TypeError in astropy ≥5.x.
    # Put APASS first: it has direct B, V, and filter_lookup for R and I,
    # making it the most reliable first choice for standard optical filters.
    exclude = {'SDSS'}
    preferred = ['APASS']
    others = [n for n in known_catalogs.keys() if n not in exclude and n not in preferred]
    catalog_names = [n for n in preferred if n in known_catalogs] + others
    if not catalog_names:
        return None, (
            "No stored field_cal presets found and no catalog plugins registered; "
            "field calibration cannot proceed"
        )

    logger.info(
        "No stored field_cal presets found. "
        "Available catalog plugins (from known_catalogs registry, not hard-coded): %s",
        catalog_names,
    )

    # Build a FieldCal using discovered catalog names plus explicit calibration
    # parameters specified by the user.
    fc = FieldCal()
    fc.catalogs               = catalog_names
    fc.source_match_tol       = FCAL_SOURCE_MATCH_TOL
    fc.source_inclusion_percent = FCAL_SOURCE_INCL_PCT
    fc.min_snr                = FCAL_MIN_SNR
    return fc, ""


def run_legacy_field_calibration(file_id: int) -> tuple:
    """
    Run photometric field calibration using the legacy FieldCalJob plugin.

    FieldCalJob is configured with:
      - field_cal: catalogs from registry + FCAL_* parameters
      - source_extraction_settings: specified SE_* parameters
      - photometry_settings: specified PHOT_* parameters

    Returns (result_data_list_or_None, meta_dict).
    """
    field_cal, fail_reason = build_default_field_cal(None)
    if field_cal is None:
        raise ValueError(fail_reason)

    job = FieldCalJob()
    job.user_id    = None
    job.session_id = None
    job.id         = str(uuid.uuid4())
    job.state      = JobState()
    job.state.status = 'in_progress'
    job.result     = FieldCalJobResult()
    job._task      = MockTask()
    job.file_ids   = [file_id]
    job.field_cal  = field_cal
    job.source_extraction_settings = make_source_extraction_settings()
    job.photometry_settings        = make_phot_settings()

    job.run()

    meta = {
        'errors':   job.result.errors,
        'warnings': job.result.warnings,
    }
    result_data = list(getattr(job.result, 'data', []))
    if job.result.errors and not result_data:
        return None, meta
    return result_data or None, meta


# ─────────────────────────────────────────────────────────────────────────────
# WCS header extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_wcs_header_fields(fits_path: Path) -> dict:
    """
    Extract WCS values from the final output FITS header.
    Supports both CD matrix and PC/CDELT representations.
    Computes effective CD matrix values when PC/CDELT is used.
    Returns None for any value that cannot be found.
    """
    out = {k: None for k in (
        'wcs_crval1', 'wcs_crval2', 'wcs_crpix1', 'wcs_crpix2',
        'wcs_cd1_1', 'wcs_cd1_2', 'wcs_cd2_1', 'wcs_cd2_2',
        'wcs_ctype1', 'wcs_ctype2', 'wcs_header_written',
    )}
    try:
        with pyfits.open(str(fits_path), memmap=False) as hdul:
            hdr = hdul[0].header
            out['wcs_crval1'] = hdr.get('CRVAL1')
            out['wcs_crval2'] = hdr.get('CRVAL2')
            out['wcs_crpix1'] = hdr.get('CRPIX1')
            out['wcs_crpix2'] = hdr.get('CRPIX2')
            out['wcs_ctype1'] = hdr.get('CTYPE1')
            out['wcs_ctype2'] = hdr.get('CTYPE2')

            if 'CD1_1' in hdr:
                # Direct CD matrix
                out['wcs_cd1_1'] = hdr.get('CD1_1')
                out['wcs_cd1_2'] = hdr.get('CD1_2', 0.0)
                out['wcs_cd2_1'] = hdr.get('CD2_1', 0.0)
                out['wcs_cd2_2'] = hdr.get('CD2_2')
            elif 'PC1_1' in hdr and 'CDELT1' in hdr:
                # PC matrix * CDELT gives effective CD
                cdelt1 = hdr.get('CDELT1', 1.0)
                cdelt2 = hdr.get('CDELT2', 1.0)
                out['wcs_cd1_1'] = hdr.get('PC1_1', 1.0) * cdelt1
                out['wcs_cd1_2'] = hdr.get('PC1_2', 0.0) * cdelt2
                out['wcs_cd2_1'] = hdr.get('PC2_1', 0.0) * cdelt1
                out['wcs_cd2_2'] = hdr.get('PC2_2', 1.0) * cdelt2

            out['wcs_header_written'] = bool(
                out['wcs_crval1'] is not None
                and out['wcs_crval2'] is not None
                and out['wcs_ctype1'] is not None
            )
    except Exception as exc:
        logger.debug("Could not read WCS from %s: %s", fits_path, exc)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Field calibration result extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_field_cal_values(fc_results: list, output_fits_path: Path) -> dict:
    """
    Extract calibration values from FieldCalResult objects.

    Primary source: FieldCalResult.zero_point_corr (the fitted zero point from
    calc_solution — the AUTHORITATIVE calibration value).
    Fallback: FITS header PHOT_M0 / PHOT_M0E (written by the field cal job).

    PhotSettings.zero_point is NOT used here — it is only an additive
    instrument offset, not a fitted calibration result.
    """
    out = {
        'zero_point': None, 'zero_point_error': None, 'zero_point_slop': None,
        'limmag5': None, 'rej_percent': None,
        'catalog_name': None, 'filter_name': None,
        '_header_fallback': False,
    }

    if fc_results:
        r = fc_results[0]
        out['zero_point']       = getattr(r, 'zero_point_corr',  None)
        out['zero_point_error'] = getattr(r, 'zero_point_error', None)
        out['zero_point_slop']  = getattr(r, 'zero_point_slop',  None)
        out['limmag5']          = getattr(r, 'limmag5',          None)
        out['rej_percent']      = getattr(r, 'rej_percent',       None)
        phot_sources = getattr(r, 'phot_results', [])
        if phot_sources:
            out['catalog_name'] = getattr(phot_sources[0], 'catalog_name', None)
            out['filter_name']  = getattr(phot_sources[0], 'filter',       None)

    # Fallback to FITS header if result fields are missing
    if out['zero_point'] is None and output_fits_path and output_fits_path.exists():
        try:
            with pyfits.open(str(output_fits_path), memmap=False) as hdul:
                hdr = hdul[0].header
                m0  = hdr.get('PHOT_M0')
                if m0 is not None:
                    out['zero_point']       = m0
                    out['zero_point_error'] = hdr.get('PHOT_M0E')
                    out['_header_fallback'] = True
        except Exception:
            pass

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-file processing
# ─────────────────────────────────────────────────────────────────────────────

def build_initial_row(index: int, input_path: Path) -> dict:
    basename = input_path.name
    obs_id   = index + 1
    return {
        'input_file':          basename,
        'object_key':          f"inputs/{index:03d}-{basename}",
        'observation_asset_id': obs_id,
        'processing_run_id':   str(uuid.uuid4()),
        'output_file_id':      None,
        'output_object_key':   None,
        'run_status':          'failed',
        'failure_reason':      '',
        'exception_type':      '',
        'exception_message':   '',
        'validation_ok':       False,
        # Reduction is never attempted — this script is for WCS+phot+fieldcal only
        'reduction_attempted': False,
        'reduction_ok':        False,
        'wcs_attempted':       False,
        'wcs_solved':          False,
        'photometry_attempted': False,
        'photometry_ok':        False,
        'field_cal_attempted':  False,
        'field_cal_ok':         False,
        'output_created':      False,
        'wcs_crval1': None, 'wcs_crval2': None,
        'wcs_crpix1': None, 'wcs_crpix2': None,
        'wcs_cd1_1':  None, 'wcs_cd1_2':  None,
        'wcs_cd2_1':  None, 'wcs_cd2_2':  None,
        'wcs_ctype1': None, 'wcs_ctype2': None,
        'wcs_header_written': False,
        'rej_percent':      None,
        'zero_point':       None,
        'zero_point_error': None,
        'zero_point_slop':  None,
        'limmag5':          None,
        'catalog_name':     None,
        'filter_name':      None,
        'elapsed_seconds':  None,
        'notes':            '',
    }


def process_one_file(index: int, input_path: Path) -> dict:
    """
    Run WCS calibration → aperture photometry → field calibration on one FITS.
    Any unhandled exception is caught; the batch continues after per-file failures.
    """
    row   = build_initial_row(index, input_path)
    t0    = time.monotonic()
    notes = []

    try:
        # ── Validate FITS ──────────────────────────────────────────────────
        logger.info("[%03d] Validating %s", index, input_path.name)
        with pyfits.open(str(input_path), memmap=False) as hdul:
            if hdul[0].data is None:
                raise ValueError("Primary HDU has no image data")
        row['validation_ok'] = True

        # ── Register data file with Afterglow ──────────────────────────────
        logger.info("[%03d] Registering data file", index)
        with afterglow_app.app_context():
            file_id = register_legacy_data_file(input_path)
        obs_id = row['observation_asset_id']
        row['output_file_id']      = str(file_id)
        row['output_object_key']   = (
            f"legacy/observations/{obs_id}/assets/{file_id}"
        )

        # ── WCS calibration ────────────────────────────────────────────────
        row['wcs_attempted'] = True
        logger.info("[%03d] WCS calibration (all schema defaults + ANET index)…", index)
        with afterglow_app.app_context():
            solved_id, wcs_meta = run_legacy_wcs(file_id)

        if solved_id is not None:
            row['wcs_solved'] = True
            notes.append('last_stage=wcs')
            with afterglow_app.app_context():
                fits_path = Path(get_data_file_path(None, solved_id))
            row['output_created'] = fits_path.exists()
            row.update(extract_wcs_header_fields(fits_path))
        else:
            err_detail = '; '.join(
                e.get('detail', str(e)) for e in wcs_meta.get('errors', [])
            )
            row['failure_reason'] = f"WCS failed: {err_detail}"
            notes.append(f'wcs_failed={err_detail[:100]}')
            logger.warning("[%03d] WCS failed: %s", index, err_detail)

        # ── Standalone aperture photometry ─────────────────────────────────
        # Default PhotSettings: mode='aperture', a=None.
        # run_photometry_job raises ValueError (missing aperture radius) when
        # mode='aperture' and a is None. This is expected; not a script bug.
        # We record the failure clearly. The zero_point from this step would
        # NOT be the fitted calibration zero point anyway (see module docstring).
        row['photometry_attempted'] = True
        logger.info("[%03d] Standalone photometry (default PhotSettings)…", index)

        # First extract sources (needed as input positions for photometry)
        sources = []
        try:
            with afterglow_app.app_context():
                stub = _StubJob()
                sources, _ = run_source_extraction_job(
                    stub, make_source_extraction_settings(), [file_id]
                )
            if stub._errors:
                raise ValueError(
                    f"Source extraction errors: {'; '.join(stub._errors[:2])}"
                )
            if not sources:
                raise ValueError("Source extraction returned no sources")
            logger.info("[%03d] Extracted %d sources", index, len(sources))
        except Exception as se_exc:
            notes.append(f'source_extraction_failed={str(se_exc)[:80]}')
            logger.info("[%03d] Source extraction failed: %s", index, se_exc)

        if sources:
            try:
                with afterglow_app.app_context():
                    run_legacy_aperture_photometry(file_id, sources)
                row['photometry_ok'] = True
                notes.append('photometry_ok')
            except Exception as phot_exc:
                row['photometry_ok'] = False
                notes.append(f'photometry_failed={str(phot_exc)[:80]}')
                logger.info("[%03d] Photometry failed: %s", index, phot_exc)
        else:
            row['photometry_ok'] = False
            notes.append('photometry_skipped=no_sources')

        # ── Field calibration ──────────────────────────────────────────────
        # Field calibration is the AUTHORITATIVE source for zero_point.
        # The fitted zero point comes from calc_solution() →
        #   FieldCalResult.zero_point_corr / FITS PHOT_M0.
        # See module docstring and extract_field_cal_values() for details.
        row['field_cal_attempted'] = True
        logger.info("[%03d] Field calibration…", index)
        try:
            with afterglow_app.app_context():
                fc_results, fc_meta = run_legacy_field_calibration(file_id)

            if fc_results:
                row['field_cal_ok'] = True
                # Remove old last_stage marker, replace with field_calibration
                notes = [n for n in notes if not n.startswith('last_stage=')]
                notes.append('last_stage=field_calibration')

                with afterglow_app.app_context():
                    fc_path = Path(get_data_file_path(None, file_id))

                fc_vals = extract_field_cal_values(fc_results, fc_path)
                if fc_vals.pop('_header_fallback', False):
                    notes.append('fallback_from_header_PHOT_M0')
                row.update({k: v for k, v in fc_vals.items()})
                row['output_created'] = fc_path.exists()
            else:
                errs = fc_meta.get('errors', [])
                err_str = '; '.join(e.get('detail', str(e)) for e in errs)
                notes.append(f'field_cal_no_results={err_str[:100]}')
                warns = fc_meta.get('warnings', [])
                warn_str = '; '.join(w.get('detail', str(w)) for w in warns) if warns else '(none)'
                logger.info("[%03d] Field cal: no results. Errors: %s | Warnings: %s",
                            index, err_str, warn_str)

        except Exception as fc_exc:
            row['field_cal_ok'] = False
            notes.append(f'field_cal_failed={str(fc_exc)[:120]}')
            logger.info("[%03d] Field cal failed: %s", index, fc_exc, exc_info=True)

        # ── Determine overall run_status ───────────────────────────────────
        if row['wcs_solved'] and row['field_cal_ok']:
            row['run_status'] = 'completed'
        elif row['wcs_solved']:
            row['run_status'] = 'partial'
        # else: remains 'failed'

    except Exception as exc:
        row['run_status']       = 'failed'
        row['exception_type']   = type(exc).__name__
        row['exception_message'] = str(exc)
        row['failure_reason']   = row['failure_reason'] or str(exc)[:150]
        logger.error(
            "[%03d] Unhandled exception processing %s",
            index, input_path.name, exc_info=True,
        )

    row['elapsed_seconds'] = round(time.monotonic() - t0, 3)
    row['notes']           = '; '.join(notes)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(rows: list) -> None:
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV written: %s (%d rows)", OUTPUT_CSV, len(rows))


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    configure_logging()
    logger.info("=" * 70)
    logger.info("test-optical.py — legacy Afterglow optical stack batch test")
    logger.info("=" * 70)
    logger.info("INPUT_DIR : %s", INPUT_DIR)
    logger.info("OUTPUT_CSV: %s", OUTPUT_CSV)
    logger.info("TMPDIR    : %s", TMPDIR)
    logger.info("ANET      : %s", ANET_INDEX_PATH)

    validate_anet_indexes(ANET_INDEX_PATH)

    # Clear and recreate per-run working directory
    reset_tmpdir()

    # Reset the Afterglow database for a clean run.
    # The SQLite DB file persists across resets (it's outside TMPDIR) but we
    # drop and recreate all tables so data files from previous runs don't
    # interfere with file_id assignments.
    logger.info("Resetting Afterglow database…")
    with afterglow_app.app_context():
        db.drop_all()
        db.create_all()
        data_root = get_root(None)
        os.makedirs(data_root, exist_ok=True)
        logger.info("Data file root: %s", data_root)

    fits_files = iter_fits_files()
    rows = []

    for index, input_path in enumerate(fits_files):
        logger.info("")
        logger.info(
            "── [%03d/%03d] %s ──",
            index + 1, len(fits_files), input_path.name,
        )
        row = process_one_file(index, input_path)
        rows.append(row)
        logger.info(
            "[%03d] %s | wcs=%-5s phot=%-5s fcal=%-5s | %.3fs | %s",
            index,
            row['run_status'],
            row['wcs_solved'],
            row['photometry_ok'],
            row['field_cal_ok'],
            row['elapsed_seconds'],
            row['notes'][:80],
        )

    write_csv(rows)

    n = len(rows)
    n_wcs  = sum(1 for r in rows if r['wcs_solved'])
    n_phot = sum(1 for r in rows if r['photometry_ok'])
    n_fcal = sum(1 for r in rows if r['field_cal_ok'])
    logger.info("")
    logger.info(
        "=== Summary: %d file(s) | WCS %d/%d | Phot %d/%d | FieldCal %d/%d ===",
        n, n_wcs, n, n_phot, n, n_fcal, n,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
