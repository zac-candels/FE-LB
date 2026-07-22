from pathlib import Path
import shutil 

def create_output_directory(dt, h, name="outputDir"):
    
    workdir = Path.cwd()

    out_dir = workdir / f"{name}_dt{dt:.2e}_h{h:.2f}"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir