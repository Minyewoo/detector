use std::rc::Rc;
use opencv::{prelude::Mat, imgproc};
use super::layer::Layer;

pub struct CannyEdgeDetection {
    layer: Box<dyn Layer>,
    treshold1: f64,
    treshold2: f64,
    aperture_size: i32,
}

impl CannyEdgeDetection {
    pub fn new(layer: Box<dyn Layer>, treshold1: f64, treshold2: f64, aperture_size: i32) -> Self {
        CannyEdgeDetection { layer, treshold1, treshold2, aperture_size }
    }
}

impl Layer for CannyEdgeDetection {
    fn process(&mut self) -> Result<Rc<Mat>, String> {
        match self.layer.process() {
            Ok(frame) => {
                let mut edges_frame = Mat::default();
                imgproc::canny(frame.as_ref(), &mut edges_frame, self.treshold1, self.treshold2, self.aperture_size, false)
                    .map_err(|err| format!("[CannyLayer] canny: {err}"))?;
                Ok(Rc::new(edges_frame))
            },
            Err(error) => Err(error),
        }
    }
}