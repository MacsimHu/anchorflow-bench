"""Assigned Case30 Knockout/MeanToken controls; never update a live Unit checkout."""
import sys

from case30_matched import ROOT, digest, main, read


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("Knockout", "MeanToken", "IntactMeanToken"):
        raise SystemExit("Usage: python case30_extra_controls.py {Knockout,MeanToken,IntactMeanToken} [matched-runner options]")
    if "--execute" in sys.argv:
        qualification = read(ROOT / "CASE30_EXTRA_QUALIFICATION.json")
        if qualification.get("passed") is not True:
            raise RuntimeError("The changed controls have no passing qualification.")
        for name, expected in qualification["sources"].items():
            if digest(ROOT / name) != expected:
                raise ValueError("A qualified extra-control source changed: " + name)
    options = dict(allocation_path=ROOT / "CASE30_EXTRA_ALLOCATION.json", variant=sys.argv[1])
    if sys.argv[1] == "IntactMeanToken":
        options.update(variant="MeanToken", reference_path=ROOT / "case30_intact_reference" / "FULL_REFERENCE.json")
    raise SystemExit(main(sys.argv[2:], **options))
