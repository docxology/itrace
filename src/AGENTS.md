# src/ — Agent Notes

Single package root: `itrace/` (the analysis core). Nothing else lives here.
No cross-project imports; the package must stay installable standalone
(`py.typed`, pure NumPy/SciPy kernel).
