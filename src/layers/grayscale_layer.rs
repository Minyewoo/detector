use std::sync::Arc;
use opencv::{prelude::Mat, imgproc};
use super::layer::Layer;

pub struct GrayscaleLayer {
    layer: Box<dyn Layer>,
}

impl GrayscaleLayer {
    pub fn new(layer: Box<dyn Layer>) -> Self {
        GrayscaleLayer { layer }
    }
}

impl Layer for GrayscaleLayer {
    fn process(&mut self) -> Result<Arc<Mat>, String> {
        match self.layer.process() {
            Ok(frame) => {
                let mut grayscale_frame = Mat::default();
                imgproc::cvt_color(frame.as_ref(), &mut grayscale_frame, imgproc::COLOR_BGR2GRAY, 0)
                    .map_err(|err: opencv::Error| format!("[GrayscaleLayer] cvt_color: {err}"))?;
                Ok(Arc::new(grayscale_frame))
            },
            Err(error) => Err(error),
        }
    }
}