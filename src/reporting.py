import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
from matplotlib.backends.backend_pdf import PdfPages
import time

def generate_final_report(processor, session_duration):
    if not processor.all_hr_values:
        return "No data collected", None, None, None
    hr_array = np.array(processor.all_hr_values)
    bp_array = np.array(processor.all_bp_values)
    report = {
        "session_duration": processor.accumulated_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hr_avg": np.mean(hr_array),
        "systolic_avg": np.mean(bp_array[:, 0]),
        "diastolic_avg": np.mean(bp_array[:, 1]),
        "current_hr": processor.all_hr_values[-1],
        "current_systolic": processor.all_bp_values[-1][0],
        "current_diastolic": processor.all_bp_values[-1][1]
    }
    df = pd.DataFrame({
        'Timestamp': processor.all_timestamps,
        'Heart_Rate': processor.all_hr_values,
        'Systolic': [bp[0] for bp in processor.all_bp_values],
        'Diastolic': [bp[1] for bp in processor.all_bp_values]
    })

    # Figure for Streamlit display
    display_fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(processor.all_timestamps, processor.all_hr_values, 'g-', linewidth=2)
    ax1.set_title("Heart Rate Over Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Heart Rate (BPM)")
    ax1.grid(True)
    ax2.plot(processor.all_timestamps, [bp[0] for bp in processor.all_bp_values], 'r-', label="Systolic")
    ax2.plot(processor.all_timestamps, [bp[1] for bp in processor.all_bp_values], 'b-', label="Diastolic")
    ax2.set_title("Blood Pressure Over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Blood Pressure (mmHg)")
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()

    # PDF report
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        pdf_fig = plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.97, "VIRTUON AI - Vital Signs Report", ha='center', fontsize=16, weight='bold', color='darkblue')
        plt.text(0.1, 0.90, f"Monitoring Period: {report['timestamp']}", fontsize=10)
        table_data = [
            ["Parameter", "Your Value", "Normal Range"],
            ["Heart Rate", f"{report['current_hr']:.1f} BPM", "60-100 BPM"],
            ["Blood Pressure", f"{report['current_systolic']:.1f}/{report['current_diastolic']:.1f} mmHg", "90-120/60-80 mmHg"]
        ]
        table = plt.table(cellText=table_data, loc='center', cellLoc='center', bbox=[0.1, 0.40, 0.8, 0.45])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('darkblue')
            else:
                cell.set_text_props(color='black')
                cell.set_facecolor('lightgray' if row % 2 else 'white')
        plt.text(0.1, 0.02, "Disclaimer: For research and educational use only.", fontsize=8, style='italic', color='gray')
        plt.axis('off')
        pdf.savefig(pdf_fig)
        plt.close(pdf_fig)
        pdf.savefig(display_fig)
        plt.close(display_fig)

    pdf_buffer.seek(0)
    return report, display_fig, df, pdf_buffer