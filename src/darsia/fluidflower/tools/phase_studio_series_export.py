"""Export already completed saved mass products without starting inference."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
from darsia.fluidflower.saved_mass_rendering import read_result, encode, write


def render_overflow(path, rows, title):
    """Plot recorded image and gas-reference masses."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(11, 4.5), layout='constrained'); FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.plot([r['elapsed_h'] for r in rows], [r['gas_mass_g'] for r in rows], label='Image-based free gas')
    matched = [r for r in rows if r.get('gas_reference')]
    if matched:
        ax.errorbar([r['gas_reference']['time_h'] for r in matched],
            [r['gas_reference']['mass_g'] for r in matched],
            yerr=[r['gas_reference']['sigma_g'] for r in matched], fmt='o', label='Overflow-derived gas reference')
    ax.set(xlabel='Time since injection start · h', ylabel='Free CO2 mass · g', title=title+' · measured gas reference')
    ax.legend(); ax.grid(alpha=.2); fig.savefig(path, dpi=120)


def export_csv(result, path):
    """Export exact recorded mass scalars to CSV."""
    columns = ['elapsed_h','injected_mass_g','aqueous_mass_g','gas_mass_g','total_mass_g',
               'gas_mass_min_g','gas_mass_max_g','dilution_factor','gas_saturation','calibration_role']
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in result['frames'])


def export(result_path, destination, kind):
    """Export a saved movie, mass table or portable result bundle."""
    path, destination = Path(result_path), Path(destination)
    if destination.exists(): raise FileExistsError('Choose a new export path')
    result = read_result(path)
    if kind == 'movie': return encode(path, destination)
    if kind == 'csv': return export_csv(result, destination)
    if kind == 'bundle':
        destination.mkdir(parents=True)
        for row in result['frames']:
            for key in ('preview','fields'): shutil.copy2(row[key], destination/Path(row[key]).name)
        for key in ('mass_curve','overflow_curve'):
            if result.get(key): shutil.copy2(result[key], destination/Path(result[key]).name)
        portable = json.loads(json.dumps(result))
        for row in portable['frames']:
            for key in ('preview','fields'): row[key] = Path(row[key]).name
        for key in ('mass_curve','overflow_curve'):
            if portable.get(key): portable[key] = Path(portable[key]).name
        if result.get('result_source_manifest'):
            shutil.copy2(result['result_source_manifest'],destination/'result_sources.json')
            portable['result_source_manifest']='result_sources.json'
        portable['series_calibration']='calibration.json'
        write(destination/'result.json', portable)
        export_csv(result, destination/'mass_per_frame.csv')
        write(destination/'calibration.json', result['calibration'])
        return
    raise ValueError('Unknown export kind')


def main(argv=None):
    """Run only the saved-product export branch of the review CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--export", choices=["movie", "csv", "bundle"])
    args = parser.parse_args(argv)
    if args.export:
        if not args.result or not args.destination:
            parser.error("Export needs result and destination")
        return export(args.result, args.destination, args.export)
    parser.error("Specify export")


if __name__ == "__main__":
    main()
