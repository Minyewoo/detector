use std::sync::Arc;
use opencv::{prelude::Mat, imgproc, core::BORDER_DEFAULT};
use super::layer::Layer;

pub struct BilateralLayer {
    layer: Box<dyn Layer>,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
}

impl BilateralLayer {
    pub fn new(layer: Box<dyn Layer>, d: i32, sigma_color: f64, sigma_space: f64) -> Self {
        BilateralLayer { layer, d, sigma_color, sigma_space }
    }
}

impl Layer for BilateralLayer {
    fn process(&mut self) -> Result<Arc<Mat>, String> {
        match self.layer.process() {
            Ok(frame) => {
                let mut smoothed_frame = Mat::default();
                imgproc::bilateral_filter(frame.as_ref(), &mut smoothed_frame, self.d, self.sigma_color, self.sigma_space, BORDER_DEFAULT)
                    .map_err(|err| format!("[BilateralLayer] bilateral_filter: {err}"))?;
                Ok(Arc::new(smoothed_frame))
            },
            Err(error) => Err(error),
        }
    }
}