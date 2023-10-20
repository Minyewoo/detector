use std::sync::Arc;
use opencv::{prelude::Mat, imgproc};
use super::layer::Layer;

pub struct CannyLayer {
    layer: Box<dyn Layer>,
    treshold1: f64,
    treshold2: f64,
    aperture_size: i32,
}

impl CannyLayer {
    pub fn new(layer: Box<dyn Layer>, treshold1: f64, treshold2: f64, aperture_size: i32) -> Self {
        CannyLayer { layer, treshold1, treshold2, aperture_size }
    }
}

impl Layer for CannyLayer {
    fn process(&mut self) -> Result<Arc<Mat>, String> {
        match self.layer.process() {
            Ok(frame) => {
                let mut edges_frame = Mat::default();
                imgproc::canny(frame.as_ref(), &mut edges_frame, self.treshold1, self.treshold2, self.aperture_size, false)
                    .map_err(|err| format!("[CannyLayer] canny: {err}"))?;
                Ok(Arc::new(edges_frame))
            },
            Err(error) => Err(error),
        }
    }
}