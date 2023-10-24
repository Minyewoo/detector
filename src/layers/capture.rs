use std::rc::Rc;
use opencv::{prelude::Mat, videoio::{VideoCapture, VideoCaptureTrait}};
use super::layer::Layer;

pub struct Capture {
    capture: Box<VideoCapture>,
}

impl Capture {
    pub fn new(capture: Box<VideoCapture>) -> Self {
        Capture { 
            capture
        }
    }
}

impl Layer for Capture {
    fn process(&mut self) -> Result<Rc<Mat>, String> {
        let mut frame = Mat::default();
        self.capture.read(&mut frame)
            .map_err(|err| format!("[CaptureLayer] failed to read a frame from VideoCaprute: {err}"))?;
        Ok(Rc::new(frame))
    }
}