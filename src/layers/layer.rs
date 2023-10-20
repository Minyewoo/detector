use std::sync::Arc;
use opencv::prelude::Mat;

pub trait Layer {
    fn process(&mut self) -> Result<Arc<Mat>, String>;
}