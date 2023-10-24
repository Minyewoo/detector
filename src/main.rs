pub mod layers;
use std::rc::Rc;
use layers::{capture_layer::CaptureLayer, grayscale_layer::GrayscaleLayer, bilateral_layer::BilateralLayer, lut_layer::LutLayer, canny_layer::CannyLayer, layer::Layer};
use opencv::{highgui, prelude::*, imgproc::{self, LINE_8}, core::{Scalar, Vector, Point, CV_8U}, imgcodecs::IMREAD_GRAYSCALE, videoio::{self, CAP_PROP_FPS},};

fn contours_matching() -> Result<(), String> {
    let name = "test";
    highgui::named_window(name, highgui::WINDOW_GUI_NORMAL | highgui::WINDOW_KEEPRATIO | highgui::WINDOW_AUTOSIZE)
        .map_err(|err| format!("[main] failed to create window: {err}"))?;
    let grayscale_template = opencv::imgcodecs::imread("/home/minyewoo/Development/opencv-object-detector/templates/bold.jpg", IMREAD_GRAYSCALE)
        .map_err(|err| format!("[main] template read: {err}"))?;
    let mut edges = Mat::default();
    imgproc::canny(&grayscale_template, &mut edges, 0.0, 255.0, 3, false)
        .map_err(|err| format!("[main] canny: {err}"))?;
    let mut contours = Vector::<Vector<Point>>::default();
    imgproc::find_contours(&edges, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::default())
        .map_err(|err| format!("[main] find_contours template: {err}"))?;
    let template_contour = contours.get(0)
        .map_err(|err| format!("[main] contours get first: {err}"))?;
    let template_moments = imgproc::moments(&template_contour, false)
        .map_err(|err| format!("[main] template moments: {err}"))?;
    let mut template_hu_moments: [f64; 7] = Default::default();
    imgproc::hu_moments(template_moments, &mut template_hu_moments)
        .map_err(|err| format!("[main] template hu_moments: {err}"))?;
    let mut look_up_table = Mat::default();

    make_look_up_table(&mut look_up_table, 0.5)?;
    let rc_look_up_table = Rc::new(look_up_table);
    let mut capture = videoio::VideoCapture::new(0, videoio::CAP_ANY)
            .map_err(|err| format!("[main] failed to capture video from  main camera: {err}")).unwrap();
    capture.set(CAP_PROP_FPS, 30.0).unwrap();
    let capture_box = Box::new(capture);
    let mut frame_pipeline = 
    CannyLayer::new(
        Box::new(
            BilateralLayer::new(
                Box::new(
                    LutLayer::new(
                        Box::new(
                            GrayscaleLayer::new(
                                Box::new(
                                    CaptureLayer::new(
                                        capture_box
                                    ),
                                )
                            ),
                        ),
                        rc_look_up_table.clone(),
                    )
                ),
                5, 
                75.0, 
                75.0,
            ),
        ), 
        70.0, 
        20.0, 
        3,
    );
    loop {
        let mut colored_frame = Mat::default();
        let mut frame_rc = frame_pipeline.process()?;
        let frame = Rc::get_mut(&mut frame_rc).unwrap();
        imgproc::find_contours(frame, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::default())
            .map_err(|err| format!("[main] find_contours: {err}"))?;
        let matched_contour = contours
            .iter()
            .filter(|contour| {
                let area = imgproc::contour_area(contour, false).unwrap();
                area > 450.0 && area < 15000.0
            })
            .map(|contour| {
                let moments = imgproc::moments(&contour, false)
                    .map_err(|err| format!("[main] template moments: {err}")).unwrap();
                let mut hu_moments: [f64; 7] = Default::default();
                imgproc::hu_moments(moments, &mut hu_moments)
                    .map_err(|err| format!("[main] template hu_moments: {err}")).unwrap();
                // let similarity = imgproc::match_shapes(&template_contour, &contour, imgproc::CONTOURS_MATCH_I2, 0.0).unwrap();
                let similarity = euclidean_distance(&template_hu_moments, &hu_moments);
                (similarity, contour)
            })
            .fold((0.095, Vector::<Point>::default()), |min_contour, contour| {
                if contour.0 < min_contour.0 {
                    println!("{}", contour.0);
                    contour
                } else {
                    min_contour
                }
            }).1;
        if matched_contour.len() > 0 {
            let mut matched_contours = Vector::<Vector<Point>>::default();
            matched_contours.push(matched_contour);
            imgproc::draw_contours(frame, &matched_contours, -1, Scalar::new(0.0, 255.0, 0.0, 255.0), 2, LINE_8, &Mat::default(), 0, Point::new(0, 0))
                .map_err(|err| format!("[main] draw_contours: {err}"))?;
        }
        highgui::imshow(name, frame)
            .map_err(|err| format!("[main] failed to display an image in the window: {err}"))?;
        let key = highgui::wait_key(1)
            .map_err(|err| format!("[main] highgui::wait_key error: {err}"))?;
        if key == 'q' as i32 {
            break;
        }
    }
    highgui::destroy_window(name)
        .map_err(|err| format!("[main] failed to destroy window: {err}"))?;
    Ok(())
}

fn euclidean_distance<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    let mut sum = 0.0;
    for i in 0..N {
        sum += (a[i] - b[i]).powf(2.0);
    }
    sum.sqrt()
}

fn make_look_up_table(look_up_table: &mut Mat, gamma: f64) -> Result<(), String> {
    *look_up_table = Mat::new_nd_vec_with_default(&Vector::from_slice(&[1,256]), CV_8U, Scalar::new(0.0, 0.0, 0.0, 0.0))
        .map_err(|err| format!("[make_look_up_table] new_nd_vec_with_default: {err}"))?;
    for i in 0..256 {
        *look_up_table.at_2d_mut(0, i).unwrap() =  ((i as f64 / 255.0).powf(gamma) * 255.0).clamp(0.0, 255.0) as u8;
    }
    Ok(())
}

fn main() -> Result<(), String> {
    contours_matching()
}
