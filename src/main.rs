use opencv::{highgui, videoio,  prelude::*,};

fn main() -> Result<(), String> {
    let name = "test";
    highgui::named_window(name, highgui::WINDOW_GUI_NORMAL | highgui::WINDOW_KEEPRATIO | highgui::WINDOW_AUTOSIZE)
        .map_err(|_| "[main] failed to create window")?;
    let mut capture = videoio::VideoCapture::new(0, videoio::CAP_ANY)
        .map_err(|_| "[main] failed to capture video from  main camera")?;
    capture.set(videoio::CAP_PROP_FPS, 30f64)
        .map_err(|_| "[main] failed to set fps cap")?;
    let mut frame = Mat::default();
    loop {
        capture.read(&mut frame)
            .map_err(|_| "[main] failed to read a frame from VideoCaprute")?;
        highgui::imshow(name, &frame)
            .map_err(|_| "[main] failed to display an image in the window")?;
        let key = highgui::wait_key(1)
            .map_err(|err| format!("[main] highgui::wait_key error: {err}"))?;
        if key == 'q' as i32 {
            break;
        }
    }
    highgui::destroy_window(name)
        .map_err(|_| "[main] failed to destroy window")?;
    Ok(())
}
