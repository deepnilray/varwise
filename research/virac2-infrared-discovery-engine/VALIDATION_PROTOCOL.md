# Independent validation and release protocol (NOT YET EXECUTED)

## Freeze before data access
1. Create a named, source- and spatially-disjoint sky-tile registry, excluding all 138 development groups plus previously opened VIRAC2 v7/v10 validation and pilot regions. Preserve a dated SHA-256 manifest and publish the definition before downloading any validation flux.
2. Predefine strict positive eligibility (real published, image-confirmed events) and negative cohorts: quiet stars, genuine non-lensing variables, photometric blends, detector-edge/saturated-wing false detections, image artifacts, duplicates, and ambiguous/abstention cases. Blind adjudicators should not see the model scores.
3. Seal the model binary hash, code hash, feature extractor, score threshold, data selection, deduplication, and post-hoc review rules. After looking at independent evaluation, any change requires **a completely new untouched field**.

## Performance measurements required
- Complete confusion matrix with denominators for *all classes*, not only published variables; Wilson confidence intervals, alert precision and recall, fraction abstained and false-leads-per-100,000 real sources.
- Ranked precision at top 10, 30 and 100, human inspection minutes per true detection, and fraction of artifacts among those top leads.
- Source leakage checks across sky tiles, match radius and shared survey exposures; event recurrence/alias control and positive-label publication-grade confidence.
- Spatial and magnitude stratification, exposure quality, blending/crowding, event timescale, seasonal gaps and possible catalog-dependent biases.
- Pairwise comparison to a reproduced published VVV baseline on the *same* untouched sky coverage and identical ground truth. Any superiority claim must state paired uncertainty and the metric being improved.

## Mandatory per-candidate evidence before calling anything new
- Original and reference FITS cutouts from multiple independent observations; astrometrically stable point-source position, no bright-wing/edge/spike/cosmic-ray artifact, no blending with nearby variable.
- Physically reasonable PSPL fit and alternatives (nova/outburst/flare/YSO, periodic variable, camera defect), with posterior uncertainties. Consider parallax/blending/finite-source effects when scientifically justified.
- Cross-match SIMBAD, VizieR, VSX, OGLE, VVV published catalogs, Gaia variability and appropriate alert databases using uncertainty-aware positions and literature dates. Archive query logs.
- Present light curve, sky coordinates, source identifier, time-scale, uncertainty, negative evidence, images and full software/campaign provenance to external astronomers for confirmation.

## Release gate
**Default remains `discovery_validated=false`.** No script in this repository is allowed to flip the gate automatically based on development metrics or an unvetted candidate. Real-world validated promotion requires an independently archived validation report and expert review. The model score is not a discovery probability.
