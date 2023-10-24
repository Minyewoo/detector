use std::rc::Rc;
use opencv::prelude::Mat;

pub trait Layer {
    fn process(&mut self) -> Result<Rc<Mat>, String>;
}