# ZTF candidate review — measured data only

**No astronomical discovery has been established by this report.**

ZTF raw observations: `data/ztf_dr24_field_298_29_45arcsec.json`
Raw SHA256: `234ce0f44115d1f3571dc062e1ceb2fda6fed7e30290f07a47e0b31585b47036`
Sky field tag: `original_45arcsec_field`

## What the checks found

- KNOWN_IN_CHECKED_CATALOG: **4**
- NOT_READY_FOR_DISCOVERY_CLAIM: **121**

## What the automatic checks mean

Each recorded ZTF measurement stays as observed. Periodic signals must exceed the reported measurement noise, appear in both time halves, and agree between independent filters. Close neighbors and peaks near the day/night observing rhythm block automatic promotion. Three original archived catalogs are checked, but catalog absence is **never** proof of discovery.

### Strongest remaining review cases

Only human-review priorities, never named or confirmed discoveries.

No object cleared every conservative screening gate in this field.

### Reviewable leads that did NOT pass all checks

These are **not** discoveries and may be uninteresting or spurious. The list is only a starting point for examining original sky images.

- **ZTF OID 686303400124632**, RA 297.999390°, Dec 29.871759° — blocked by: no_very_strong_periodogram_peak, not_independently_confirmed_in_two_filters. [Inspect image](https://aladin.cds.unistra.fr/AladinLite/?target=297.9993896484375%2029.87175941467285&fov=0.03&survey=P%2FDSS2%2Fcolor).
- **ZTF OID 686103400089985**, RA 298.005829°, Dec 29.869354° — blocked by: not_independently_confirmed_in_two_filters, another_ztf_source_within_3_arcsec_blending_risk. [Inspect image](https://aladin.cds.unistra.fr/AladinLite/?target=298.0058288574219%2029.869354248046875&fov=0.03&survey=P%2FDSS2%2Fcolor).
- **ZTF OID 686303400172784**, RA 298.007416°, Dec 29.870432° — blocked by: no_very_strong_periodogram_peak, not_independently_confirmed_in_two_filters. [Inspect image](https://aladin.cds.unistra.fr/AladinLite/?target=298.0074157714844%2029.870431900024414&fov=0.03&survey=P%2FDSS2%2Fcolor).
- **ZTF OID 686103400089683**, RA 298.002045°, Dec 29.880432° — blocked by: variability_not_clearly_above_photometric_errors, not_independently_confirmed_in_two_filters. [Inspect image](https://aladin.cds.unistra.fr/AladinLite/?target=298.0020446777344%2029.88043212890625&fov=0.03&survey=P%2FDSS2%2Fcolor).
- **ZTF OID 686303400085327**, RA 297.999817°, Dec 29.863180° — blocked by: no_very_strong_periodogram_peak, not_independently_confirmed_in_two_filters. [Inspect image](https://aladin.cds.unistra.fr/AladinLite/?target=297.99981689453125%2029.86318016052246&fov=0.03&survey=P%2FDSS2%2Fcolor).
- **ZTF OID 686303400085155**, RA 298.014008°, Dec 29.866438° — blocked by: no_very_strong_periodogram_peak, not_independently_confirmed_in_two_filters, another_ztf_source_within_3_arcsec_blending_risk. [Inspect image](https://aladin.cds.unistra.fr/AladinLite/?target=298.0140075683594%2029.866437911987305&fov=0.03&survey=P%2FDSS2%2Fcolor).
- **ZTF OID 640116100094495**, RA 298.012909°, Dec 29.863583° — blocked by: variability_not_clearly_above_photometric_errors, signal_does_not_repeat_in_both_time_halves, not_independently_confirmed_in_two_filters. [Inspect image](https://aladin.cds.unistra.fr/AladinLite/?target=298.0129089355469%2029.863582611083984&fov=0.03&survey=P%2FDSS2%2Fcolor).
- **ZTF OID 686303400124751**, RA 297.999603°, Dec 29.869246° — blocked by: no_very_strong_periodogram_peak, not_independently_confirmed_in_two_filters, another_ztf_source_within_3_arcsec_blending_risk. [Inspect image](https://aladin.cds.unistra.fr/AladinLite/?target=297.9996032714844%2029.869245529174805&fov=0.03&survey=P%2FDSS2%2Fcolor).
- **ZTF OID 686303400195127**, RA 298.004211°, Dec 29.866604° — blocked by: no_very_strong_periodogram_peak, signal_does_not_repeat_in_both_time_halves, not_independently_confirmed_in_two_filters. [Inspect image](https://aladin.cds.unistra.fr/AladinLite/?target=298.00421142578125%2029.86660385131836&fov=0.03&survey=P%2FDSS2%2Fcolor).
- **ZTF OID 686303400124726**, RA 298.001617°, Dec 29.869652° — blocked by: no_very_strong_periodogram_peak, signal_does_not_repeat_in_both_time_halves, not_independently_confirmed_in_two_filters. [Inspect image](https://aladin.cds.unistra.fr/AladinLite/?target=298.0016174316406%2029.869651794433594&fov=0.03&survey=P%2FDSS2%2Fcolor).
- **ZTF OID 686303400154555**, RA 297.989929°, Dec 29.865637° — blocked by: no_very_strong_periodogram_peak, signal_does_not_repeat_in_both_time_halves, not_independently_confirmed_in_two_filters. [Inspect image](https://aladin.cds.unistra.fr/AladinLite/?target=297.98992919921875%2029.865636825561523&fov=0.03&survey=P%2FDSS2%2Fcolor).
- **ZTF OID 686303400172891**, RA 298.007965°, Dec 29.864193° — blocked by: no_very_strong_periodogram_peak, signal_does_not_repeat_in_both_time_halves, not_independently_confirmed_in_two_filters. [Inspect image](https://aladin.cds.unistra.fr/AladinLite/?target=298.0079650878906%2029.864192962646484&fov=0.03&survey=P%2FDSS2%2Fcolor).

## Before reporting any discovery

1. Inspect sky images and original ZTF observation flags (SNAD mirror    rows in this archive do **not** include ZTF `catflags`).
2. Check SIMBAD, Gaia source astrometry, recent VSX updates, other    survey catalogs, and neighboring/blended objects individually.
3. Confirm the period against aliases and independent survey images    or follow-up measurements; classify without relying on three-class ML guesses.
4. Have an astronomer independently review the evidence before any    submission or public claim. The current pipeline has **no measured    real-world false-alarm rate**.

## Technical caution

The Lomb–Scargle false-alarm number assumes idealized white noise. It is **not** the measured probability of a false star discovery. The thresholds are deliberately strict exploratory screening values, not calibrated sensitivity/specificity on real control stars.

