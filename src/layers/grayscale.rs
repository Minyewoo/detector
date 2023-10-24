use std::rc::Rc;
use opencv::{prelude::Mat, imgproc};
use super::layer::Layer;

pub struct Grayscale {
    layer: Box<dyn Layer>,
}

impl Grayscale {
    pub fn new(layer: Box<dyn Layer>) -> Self {
        Grayscale { layer }
    }
}

impl Layer for Grayscale {
    fn process(&mut self) -> Result<Rc<Mat>, String> {
        match self.layer.process() {
            Ok(frame) => {
                let mut grayscale_frame = Mat::default();
                imgproc::cvt_color(frame.as_ref(), &mut grayscale_frame, imgproc::COLOR_BGR2GRAY, 0)
                    .map_err(|err: opencv::Error| format!("[GrayscaleLayer] cvt_color: {err}"))?;
                Ok(Rc::new(grayscale_frame))
            },
            Err(error) => Err(error),
        }
    }
}