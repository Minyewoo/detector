use std::sync::Arc;
use opencv::{prelude::Mat, videoio::{VideoCapture, VideoCaptureTrait, self, CAP_PROP_FPS}};
use super::layer::Layer;

pub struct CaptureLayer {
    capture: VideoCapture,
}

impl CaptureLayer {
    pub fn new() -> Self {
        let mut capture = videoio::VideoCapture::new(0, videoio::CAP_ANY)
            .map_err(|err| format!("[main] failed to capture video from  main camera: {err}")).unwrap();
        capture.set(CAP_PROP_FPS, 30.0).unwrap();
        CaptureLayer { 
            capture
        }
    }
}

impl Layer for CaptureLayer {
    fn process(&mut self) -> Result<Arc<Mat>, String> {
        let mut frame = Mat::default();
        self.capture.read(&mut frame)
            .map_err(|err| format!("[CaptureLayer] failed to read a frame from VideoCaprute: {err}"))?;
        Ok(Arc::new(frame))
    }
}