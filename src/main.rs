use opencv::{highgui, videoio,  prelude::*, imgproc::{self, LINE_8}, core::{Scalar, Vector, Rect, Size, convert_scale_abs, KeyPoint, DMatch, NORM_HAMMING, Point, BORDER_DEFAULT, lut, Vec2i, CV_8U, Size2i}, objdetect::CascadeClassifier, imgcodecs::IMREAD_GRAYSCALE, features2d::{SIFT, BFMatcher, draw_matches_knn, DrawMatchesFlags, ORB, ORB_ScoreType, draw_matches},};

// fn main() -> Result<(), String> {
//     let name = "test";
//     highgui::named_window(name, highgui::WINDOW_GUI_NORMAL | highgui::WINDOW_KEEPRATIO | highgui::WINDOW_AUTOSIZE)
//         .map_err(|err| format!("[main] failed to create window: {err}"))?;
//     let mut capture = videoio::VideoCapture::new(0, videoio::CAP_ANY)
//         .map_err(|err| format!("[main] failed to capture video from  main camera: {err}"))?;
//     capture.set(videoio::CAP_PROP_FPS, 30f64)
//         .map_err(|err| format!("[main] failed to set fps cap: {err}"))?;
//     let mut frame = Mat::default();
//     let mut grayscale = Mat::default();
//     let mut binary = Mat::default();
//     let mut smoothed = Mat::default();
//     let mut filtered = Mat::default();
//     let mut matched = Mat::default();
//     let mut normalized = Mat::default();
//     let mut tresholded = Mat::default();
//     let mut filter = Mat::from_slice_2d(&[[-1.0,-1.0,-1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]).unwrap();
//     let mut contours = Vector::<Vector::<Point>>::new();
//     let mut hierarchy = Mat::default();
//     let mut empty = Mat::default();
//     let zero_offset = Point::new(0, 0);
//     // let mut contours_frame = Mat::default();
//     let color = Scalar::new(255.0, 255.0, 255.0, 255.0);
//     let template = opencv::imgcodecs::imread("/home/minyewoo/Development/opencv-object-detector/templates/cross.png", IMREAD_COLOR)
//         .map_err(|err| format!("[main] template read: {err}"))?;
//     let mut copy = template.clone();
//     loop {
//         capture.read(&mut frame)
//             .map_err(|err| format!("[main] failed to read a frame from VideoCaprute: {err}"))?;
//         imgproc::match_template(&template, &template, &mut matched, imgproc::TM_CCOEFF, &empty)
//             .map_err(|err| format!("[main] match_template: {err}"))?;
//         normalize(&matched, &mut normalized, 0.0, 255.0, NORM_MINMAX, CV_8UC1, &empty)
//             .map_err(|err| format!("[main] normalize: {err}"))?;
//         imgproc::threshold(&normalized, &mut tresholded, 180.0, 255.0, imgproc::THRESH_BINARY)
//             .map_err(|err| format!("[main] failed to binaryze grayscale: {err}"))?;
//         imgproc::find_contours(&tresholded, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(template.cols() / 2, template.rows() / 2))
//             .map_err(|err| format!("find_contours: {err}"))?;
//         let kavo = contours
//             .iter()
//             .map(|contour| imgproc::moments(&contour, false))
//             .flatten()
//             .map(|moments| {
//                 let centroid = Point::new((moments.m10 / moments.m00) as i32, (moments.m01 / moments.m00) as i32);
//                 print!("({}, {})", centroid.x, centroid.y);
//                 imgproc::circle(&mut copy, centroid, 3, Scalar::new(0.0, 255.0, 0.0, 0.0), 3, LINE_8, 0)
//             }).collect::<Vec<Result<_, _>>>();
//         print!("{}", kavo.len());
//         // imgproc::cvt_color(&frame, &mut grayscale, imgproc::COLOR_BGR2GRAY, 0)
//         //     .map_err(|err| format!("[main] failed to convert frame to grayscale: {err}"))?;
//         // imgproc::threshold(&grayscale, &mut binary, 127.0, 255.0, imgproc::THRESH_BINARY)
//         //     .map_err(|err| format!("[main] failed to binaryze grayscale: {err}"))?;
//         // imgproc::gaussian_blur(&grayscale, &mut smoothed, Size2i::new(3, 3), 2.0, 2.0, BORDER_DEFAULT)
//         //     .map_err(|err| format!("[main] gaussian_blur: {err}"))?;
//         // imgproc::filter_2d(&smoothed, &mut filtered, -1, &filter, Point::new(-1, -1), 0.0, BORDER_CONSTANT)
//         //     .map_err(|err| format!("[main] filter_2d: {err}"))?;
//         // imgproc::find_contours(&binary, &mut contours, 3, 2, zero_offset)
//         //     .map_err(|err| format!("find_contours: {err}"))?;
//         // imgproc::draw_contours(&mut frame, &contours, 0, 
//         //     color, 2, LINE_AA, &hierarchy, 0, zero_offset)
//         //     .map_err(|err| format!("draw_contours: z{err}"))?;
//         highgui::imshow(name, &copy)
//             .map_err(|err| format!("[main] failed to display an image in the window: {err}"))?;
//         let key = highgui::wait_key(1)
//             .map_err(|err| format!("[main] highgui::wait_key error: {err}"))?;
//         if key == 'q' as i32 {
//             break;
//         }
//     }
//     highgui::destroy_window(name)
//         .map_err(|err| format!("[main] failed to destroy window: {err}"))?;
//     Ok(())
// }

fn haar_cascade() -> Result<(), String> {
    let name = "test";
    highgui::named_window(name, highgui::WINDOW_GUI_NORMAL | highgui::WINDOW_KEEPRATIO | highgui::WINDOW_AUTOSIZE)
        .map_err(|err| format!("[main] failed to create window: {err}"))?;
    let mut capture = videoio::VideoCapture::new(0, videoio::CAP_ANY)
        .map_err(|err| format!("[main] failed to capture video from  main camera: {err}"))?;
    capture.set(videoio::CAP_PROP_FPS, 30f64)
        .map_err(|err| format!("[main] failed to set fps cap: {err}"))?;
    let mut frame = Mat::default();
    let mut grayscale_frame = Mat::default();
    loop {
        capture.read(&mut frame)
            .map_err(|err| format!("[main] failed to read a frame from VideoCaprute: {err}"))?;
        imgproc::cvt_color(&frame, &mut grayscale_frame, imgproc::COLOR_BGR2GRAY, 0)
                .map_err(|err: opencv::Error| format!("[main] failed to convert frame to grayscale: {err}"))?;
        let mut contrast_grayscale = Mat::default();
        convert_scale_abs(&grayscale_frame, &mut contrast_grayscale, 0.4, 0.1).map_err(|err| format!("[main] convert_scale_abs: {err}"))?;
        let mut cascade = CascadeClassifier::new("/home/minyewoo/Development/opencv-object-detector/cascade.xml").unwrap();
        let mut rects = Vector::<Rect>::default();
        cascade.detect_multi_scale(&contrast_grayscale, &mut rects, 1.1, 50, 0, Size::new(60, 60), Size::new(160, 160)).unwrap();
        rects
            .into_iter()
            .for_each(|rect| {
                imgproc::rectangle(&mut frame, rect, Scalar::new(0.0, 255.0, 0.0, 255.0), 2, LINE_8, 0).unwrap();
            });
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


fn brute_force_matching_sift() -> Result<(), String> {
    let name = "test";
    highgui::named_window(name, highgui::WINDOW_GUI_NORMAL | highgui::WINDOW_KEEPRATIO | highgui::WINDOW_AUTOSIZE)
        .map_err(|err| format!("[main] failed to create window: {err}"))?;
    let mut capture = videoio::VideoCapture::new(0, videoio::CAP_ANY)
        .map_err(|err| format!("[main] failed to capture video from  main camera: {err}"))?;
    capture.set(videoio::CAP_PROP_FPS, 30f64)
        .map_err(|err| format!("[main] failed to set fps cap: {err}"))?;
    let mut frame = Mat::default();
    let mut grayscale_frame = Mat::default();
    let grayscale_template = opencv::imgcodecs::imread("/home/minyewoo/Development/opencv-object-detector/templates/cross.png", IMREAD_GRAYSCALE)
        .map_err(|err| format!("[main] template read: {err}"))?;
    let mut sift = SIFT::create(0, 3, 0.04, 10.0, 1.6).unwrap();
    let mut template_keypoints = Vector::<KeyPoint>::default();
    let mut template_descriptors = Mat::default();
    sift.detect_and_compute(&grayscale_template, &Mat::default(), &mut template_keypoints, &mut template_descriptors, false)
        .map_err(|err| format!("[main] SIFT detect_and_compute on template: {err}"))?;
    let mut frame_keypoints = Vector::<KeyPoint>::default();
    let mut frame_descriptors = Mat::default();
    let mut matcher = BFMatcher::create(NORM_HAMMING, true).unwrap();
    loop {
        capture.read(&mut frame)
            .map_err(|err| format!("[main] failed to read a frame from VideoCaprute: {err}"))?;
        imgproc::cvt_color(&frame, &mut grayscale_frame, imgproc::COLOR_BGR2GRAY, 0)
                .map_err(|err: opencv::Error| format!("[main] failed to convert frame to grayscale: {err}"))?;
        sift.detect_and_compute(&grayscale_frame, &Mat::default(), &mut frame_keypoints, &mut frame_descriptors, false)
            .map_err(|err: opencv::Error| format!("[main] SIFT detect_and_compute on frame: {err}"))?;
        let mut matches = Vector::<Vector<DMatch>>::default();
        matcher.knn_train_match(&template_descriptors, &frame_descriptors, &mut matches, 2, &Mat::default(), false)
            .map_err(|err: opencv::Error| format!("[main] BFMatcher knn_match: {err}"))?;
        let good_matches: Vector::<Vector<DMatch>> = matches
            .iter()
            .filter(|descr_matches| {
                let m = descr_matches.get(0).unwrap();
                let n = descr_matches.get(1).unwrap();
                m.distance < 0.75 * n.distance
            })
            .map(|descr_matches| Vector::from_slice(&[descr_matches.get(0).unwrap()]))
            .collect();
        let mut matching_image = Mat::default();
        draw_matches_knn(&grayscale_template, &template_keypoints, &grayscale_frame, &frame_keypoints, &good_matches, &mut matching_image, Scalar::new(0.0, 255.0, 0.0, 255.0), Scalar::all(-1.0), &Vector::<Vector<i8>>::default(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS)
            .map_err(|err: opencv::Error| format!("[main] draw_matches_knn: {err}"))?;
        highgui::imshow(name, &matching_image)
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

fn brute_force_matching_orb() -> Result<(), String> {
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
    let mut orb = ORB::create(500, 1.2, 8, 31, 0, 2, ORB_ScoreType::HARRIS_SCORE, 31, 20).unwrap();
    let mut template_keypoints = Vector::<KeyPoint>::default();
    let mut template_descriptors = Mat::default();
    orb.detect_and_compute(&grayscale_template, &Mat::default(), &mut template_keypoints, &mut template_descriptors, false)
        .map_err(|err| format!("[main] ORB detect_and_compute on template: {err}"))?;
    let mut frame_keypoints = Vector::<KeyPoint>::default();
    let mut frame_descriptors = Mat::default();
    let matcher = BFMatcher::create(NORM_HAMMING, true).unwrap();
    loop {
        capture.read(&mut frame)
            .map_err(|err| format!("[main] failed to read a frame from VideoCaprute: {err}"))?;
        imgproc::cvt_color(&frame, &mut grayscale_frame, imgproc::COLOR_BGR2GRAY, 0)
                .map_err(|err: opencv::Error| format!("[main] failed to convert frame to grayscale: {err}"))?;
        orb.detect_and_compute(&grayscale_frame, &Mat::default(), &mut frame_keypoints, &mut frame_descriptors, false)
            .map_err(|err: opencv::Error| format!("[main] SIFT detect_and_compute on frame: {err}"))?;
        let mut matches = Vector::<DMatch>::default();
        matcher.train_match(&template_descriptors, &frame_descriptors, &mut matches, &Mat::default())
            .map_err(|err: opencv::Error| format!("[main] BFMatcher knn_match: {err}"))?;
        // let mut matches = matches.to_vec().sort_by(|a, b| a.distance.total_cmp(&b.distance));
        let mask: Vector::<i8> = matches.iter().map(
            |match_| match match_.distance <= 55. {
                true => 1,
                false => 0,
            },
        ).collect();
        let zero_mask: Vector::<i8> = matches.iter().map(|_| 0).collect();
        let mut matching_image = Mat::default();
        if matches.len() >= 7 {
            draw_matches(&grayscale_template, &template_keypoints, &grayscale_frame, &frame_keypoints, &matches, &mut matching_image, Scalar::new(0.0, 255.0, 0.0, 255.0), Scalar::all(-1.0), &mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS)
            .map_err(|err: opencv::Error| format!("[main] draw_matches_knn: {err}"))?;
        } else {
            draw_matches(&grayscale_template, &Vector::<KeyPoint>::default(), &grayscale_frame, &Vector::<KeyPoint>::default(), &matches, &mut matching_image, Scalar::new(0.0, 255.0, 0.0, 255.0), Scalar::all(-1.0), &zero_mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS)
            .map_err(|err: opencv::Error| format!("[main] draw_matches_knn: {err}"))?;
        }
        highgui::imshow(name, &matching_image)
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
    let mut contrast_grayscale = Mat::default();
    let mut look_up_table = Mat::default();
    make_look_up_table(&mut look_up_table, 0.5)?;
    loop {
        capture.read(&mut frame)
            .map_err(|err| format!("[main] failed to read a frame from VideoCaprute: {err}"))?;
        imgproc::cvt_color(&frame, &mut grayscale_frame, imgproc::COLOR_BGR2GRAY, 0)
            .map_err(|err: opencv::Error| format!("[main] failed to convert frame to grayscale: {err}"))?;
        // convert_scale_abs(&grayscale_frame, &mut contrast_grayscale, 0.5, 10.0).map_err(|err| format!("[main] convert_scale_abs: {err}"))?;
        lut(&grayscale_frame, &look_up_table, &mut contrast_grayscale)
            .map_err(|err: opencv::Error| format!("[main] lut: {err}"))?;
        // imgproc::gaussian_blur(&contrast_grayscale, &mut frame_smoothed, Size2i::new(3, 3), 2.0, 2.0, BORDER_DEFAULT)
        //     .map_err(|err| format!("[main] gaussian_blur: {err}"))?;
        imgproc::bilateral_filter(&contrast_grayscale, &mut frame_smoothed, 5, 75.0, 75.0, BORDER_DEFAULT)
            .map_err(|err| format!("[main] bilateral_filter: {err}"))?;
        imgproc::canny(&frame_smoothed, &mut edges, 70.0, 200.0, 3, false)
            .map_err(|err| format!("[main] canny: {err}"))?;
        let mut contours = Vector::<Vector<Point>>::default();
        imgproc::find_contours(&edges, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::default())
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
            imgproc::draw_contours(&mut frame, &matched_contours, -1, Scalar::new(0.0, 255.0, 0.0, 255.0), 2, LINE_8, &Mat::default(), 0, Point::new(0, 0))
                .map_err(|err| format!("[main] draw_contours: {err}"))?;
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
    // haar_cascade()
    // brute_force_matching_orb()
    contours_matching()
}
