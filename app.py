import streamlit as st
from src.rppg_processor import RPPGProcessor
from src.reporting import generate_final_report
import cv2
import numpy as np
import time

def main():
    st.set_page_config(page_title="VIRTUON AI Health Monitor", page_icon="❤️", layout="wide")
    st.title("VIRTUON AI Health Monitor")
    st.markdown("Upload a video to monitor heart rate and blood pressure using rPPG technology.")

    # Session state initialization
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'session_complete' not in st.session_state:
        st.session_state.session_complete = False
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'video_uploaded' not in st.session_state:
        st.session_state.video_uploaded = None

    # Sidebar settings
    st.sidebar.header("Settings")
    session_duration = st.sidebar.slider("Session Duration (seconds)", 40, 120, 60, 5)

    # Video upload
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if video_file:
        st.session_state.video_uploaded = video_file
        st.session_state.processor = None  # Reset processor on new video upload
        st.session_state.is_running = False  # Ensure previous session is stopped

    if st.session_state.video_uploaded and not st.session_state.is_running:
        if st.button("Start Measurement"):
            st.session_state.is_running = True
            st.session_state.session_complete = False
            st.session_state.processor = RPPGProcessor(session_duration)
            st.session_state.processor.load_video(st.session_state.video_uploaded)

    # Stop button
    if st.session_state.is_running:
        if st.button("Stop Measurement"):
            st.session_state.is_running = False
            st.session_state.session_complete = True

    # Main UI layout: Video (left), Waveforms (right)
    video_col, waveform_col = st.columns([1, 1])
    with video_col:
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
    with waveform_col:
        plot_placeholder = st.empty()

    # Results area
    results_area = st.container()

    # Processing loop
    if st.session_state.is_running and st.session_state.processor:
        processor = st.session_state.processor
        try:
            while st.session_state.is_running and processor.cap.isOpened():
                ret, frame = processor.cap.read()
                if not ret:
                    status_placeholder.error("End of video or error reading frame")
                    break

                processed_frame, hr, bp, status = processor.process_frame(frame)
                plot_buf = processor.update_plot()

                frame_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                status_placeholder.text(status)
                plot_placeholder.image(plot_buf, use_container_width=True)

                if processor.accumulated_time >= session_duration:
                    st.session_state.is_running = False
                    st.session_state.session_complete = True
                    break

                time.sleep(0.03)  # Control frame rate
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")
        finally:
            if st.session_state.session_complete:
                report, fig, df, pdf = generate_final_report(processor, session_duration)
                results_area.header("Final Report")
                col1, col2 = results_area.columns(2)
                with col1:
                    st.metric("Average Heart Rate", f"{report['hr_avg']:.1f} BPM")
                with col2:
                    st.metric("Average Blood Pressure", f"{report['systolic_avg']:.1f}/{report['diastolic_avg']:.1f} mmHg")
                if fig:
                    results_area.pyplot(fig)
                if df is not None:
                    csv = df.to_csv(index=False)
                    results_area.download_button(label="Download CSV", data=csv, file_name="rppg_data.csv", mime="text/csv")
                if pdf:
                    results_area.download_button(label="Download PDF Report", data=pdf, file_name="rppg_report.pdf", mime="application/pdf")
                if results_area.button("Start New Measurement"):
                    st.session_state.is_running = False
                    st.session_state.session_complete = False
                    st.session_state.processor.cleanup()
                    del st.session_state.processor
                    st.rerun()

    st.sidebar.caption("Disclaimer: For educational use only, not for medical diagnosis.")

if __name__ == "__main__":
    main()
