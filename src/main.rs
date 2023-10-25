use std::time::Duration;

use opencv::{highgui, videoio,  prelude::*, imgproc::{self, LINE_8}, core::{Scalar, Vector, Point, BORDER_DEFAULT, lut, CV_8U, CV_16S, convert_scale_abs, Size2i, Point_}, imgcodecs::IMREAD_GRAYSCALE,};
use simpler_timer::Timer;
fn contours_matching() -> Result<(), String> {
    let name = "test";
    highgui::named_window(name, highgui::WINDOW_GUI_NORMAL | highgui::WINDOW_KEEPRATIO | highgui::WINDOW_AUTOSIZE)
        .map_err(|err| format!("[main] failed to create window: {err}"))?;
    let mut capture = videoio::VideoCapture::new(0, videoio::CAP_ANY)
        .map_err(|err| format!("[main] failed to capture video from  main camera: {err}"))?;
    capture.set(videoio::CAP_PROP_FPS, 30f64)
        .map_err(|err| format!("[main] failed to set fps cap: {err}"))?;
    let mut frame = Mat::default();
    let mut grayscale_frame = Mat::default();
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
    let mut frame_smoothed = Mat::default();
    let mut bilateral_filtered = Mat::default();
    let mut contrast_grayscale = Mat::default();
    let mut look_up_table = Mat::default();
    let mut bytes_image = Mat::default();
    let mut last_similarity = 0.0;
    let mut last_centroid = Point_::<f32>::new(0.0, 0.0);
    let similarity_reset_timer = Timer::with_duration(Duration::from_millis(800));
    let centroid_reset_timer = Timer::with_duration(Duration::from_millis(800));
    make_look_up_table(&mut look_up_table, 0.8)?;
    loop {
        if similarity_reset_timer.expired() {
            last_similarity = 0.0;
            similarity_reset_timer.reset();
        }
        if centroid_reset_timer.expired() {
            last_centroid = Point_::<f32>::new(0.0, 0.0);
            centroid_reset_timer.reset();
        }
        capture.read(&mut frame)
            .map_err(|err| format!("[main] failed to read a frame from VideoCaprute: {err}"))?;
        imgproc::cvt_color(&frame, &mut grayscale_frame, imgproc::COLOR_BGR2GRAY, 0)
            .map_err(|err: opencv::Error| format!("[main] failed to convert frame to grayscale: {err}"))?;
        // convert_scale_abs(&grayscale_frame, &mut contrast_grayscale, 0.5, 10.0).map_err(|err| format!("[main] convert_scale_abs: {err}"))?;
        lut(&grayscale_frame, &look_up_table, &mut contrast_grayscale)
            .map_err(|err: opencv::Error| format!("[main] lut: {err}"))?;
        // imgproc::gaussian_blur(&grayscale_frame, &mut frame_smoothed, Size2i::new(3, 3), 0.0, 0.0, BORDER_DEFAULT)
        //     .map_err(|err| format!("[main] gaussian_blur: {err}"))?;
        imgproc::bilateral_filter(&contrast_grayscale, &mut bilateral_filtered, 5, 75.0, 75.0, BORDER_DEFAULT)
            .map_err(|err| format!("[main] bilateral_filter: {err}"))?;
        // imgproc::laplacian(&frame_smoothed, &mut edges, CV_16S, 3, 1.0, 0.0, BORDER_DEFAULT)
        //     .map_err(|err| format!("[main] laplacian: {err}"))?;
        // convert_scale_abs(&edges, &mut bytes_image, 1.0, 0.0)
        //     .map_err(|err| format!("[main] convert_scale_abs: {err}"))?;
        imgproc::canny(&bilateral_filtered, &mut edges, 100.0, 200.0, 3, false)
            .map_err(|err| format!("[main] canny: {err}"))?;
        let mut contours = Vector::<Vector<Point>>::default();
        imgproc::find_contours(&edges, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::default())
            .map_err(|err| format!("[main] find_contours: {err}"))?;
        let matched_contour = contours
            .iter()
            .filter(|contour| {
                let area = imgproc::contour_area(contour, false).unwrap();
                // println!("{}", area);
                area > 200.0
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
            .filter(|contour| {
                let similarity = contour.0;
                match last_similarity {
                    0.0 => {
                        last_similarity = similarity;
                        true
                    },
                    _ => {
                        let similarity_distance = (last_similarity - similarity).abs();
                        if similarity < last_similarity || similarity_distance <= 0.1 {
                            last_similarity = similarity;
                            true
                        } else {
                            false
                        }
                    }
                }
            })
            .filter(|contour| {
                let centroid= imgproc::min_area_rect(&contour.1).unwrap().center;
                let centroids_distance = ((last_centroid.x - centroid.x).powf(2.0)  + (last_centroid.y - centroid.y).powf(2.0)).sqrt();
                // println!("{}", centroids_distance);
                if centroids_distance <= 70.0 || last_centroid == (Point_ { x: 0.0, y: 0.0 })  {
                    last_centroid = centroid;
                    centroid_reset_timer.reset();
                    true
                } else {
                    false
                }
            })
            .fold((0.2, Vector::<Point>::default()), |min_contour, contour| {
                if contour.0 < min_contour.0 {
                    println!("{}", contour.0);
                    contour
                } else {
                    min_contour
                }
            }).1;
        if matched_contour.len() > 0 {
            let area = imgproc::contour_area(&matched_contour, false).unwrap();
            // println!("{}", area);
            similarity_reset_timer.reset();
            centroid_reset_timer.reset();
            let mut matched_contours = Vector::<Vector<Point>>::default();
            matched_contours.push(matched_contour);
            imgproc::draw_contours(&mut frame, &matched_contours, -1, Scalar::new(0.0, 255.0, 0.0, 255.0), 2, LINE_8, &Mat::default(), 0, Point::new(0, 0))
                .map_err(|err| format!("[main] draw_contours: {err}"))?;
            imgproc::circle(&mut frame, Point::new(last_centroid.x as i32, last_centroid.y as i32), 2, Scalar::new(0.0, 0.0, 255.0, 255.0), 2, LINE_8, 0)
                .map_err(|err| format!("[main] circle: {err}"))?;
        }
        highgui::imshow(name, &frame)
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
