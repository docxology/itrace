# Graphical abstract {#sec:graphical-abstract}

The package is intentionally composed as a narrow capture shell feeding a pure
analysis core, with every live display downstream of Python-computed payloads
rather than browser-side analysis. The graphical abstract
([@fig:graphical-abstract]) uses an eye-to-code metaphor: an observed eye is
converted into typed records, the pure Python core performs the tested signal
processing, and generated reports/export artifacts carry the evidence forward.
Its badges intentionally name only the current gate, the five-session
diagnostic v1 scope, and the absence of any reference-device claim. The footer
also records the MIT license, fixed GitHub release URL, and the Zenodo DOI for
this version. The figure communicates architecture and evidence boundaries;
it is not a validation figure for webcam accuracy.

![Graphical abstract of the iTrace eye-to-code evidence pipeline. A stylized observed eye feeds typed gaze, saccade, pupil, and quality records into the pure Python core, which then produces reports, figures, and live displays from Python-computed payloads. Three badges summarise the current verification gate, the five-session diagnostic v1 release scope, and the no-reference-device-claim boundary. The footer states MIT License, github.com/docxology/itrace, and the Zenodo DOI 10.5281/zenodo.20614909; the figure describes architecture and evidence boundaries, not webcam accuracy validation.](../output/figures/graphical_abstract.png){#fig:graphical-abstract width=100%}
