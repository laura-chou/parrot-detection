import logging
import cv2
from ultralytics import YOLO

# ==========================================
# 設定 Python 內建 Logger
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==========================================
# 1. 分析單張圖片的函式
# ==========================================
def detect_image(model_path, image_path):
    logger.info(f"開始分析圖片: {image_path}")

    # 載入模型
    model = YOLO(model_path)

    # 進行預測 (conf=0.5 代表信心度超過 50% 的物件才會顯示，可減少雜物)
    results = model(image_path, conf=0.5)

    for result in results:
        result.show()  # 彈出視窗顯示結果
        result.save(filename="result_image.jpg")  # 將結果存成 result_image.jpg

    logger.info("圖片分析完成，已儲存為 result_image.jpg")


# ==========================================
# 2. 分析影片的函式
# ==========================================
def detect_video(model_path, video_path, scale=0.5):
    """分析影片函式

    :param model_path: 模型權重檔路徑
    :param video_path: 影片檔案路徑
    :param scale: 視窗顯示縮放比例 (預設 0.5 即原圖 50% 大小)
    """
    logger.info(f"開始分析影片: {video_path}")

    # 載入模型
    model = YOLO(model_path)

    # 開啟影片 (如果要用視訊鏡頭，把 video_path 改成 0)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logger.info("影片播放完畢。")
            break

        # 對當前畫面進行預測
        results = model(frame, conf=0.5)

        # 取得畫好標記框的畫面
        annotated_frame = results[0].plot()

        # 1. 將影片畫面依比例縮小，避免視窗過大
        resized_frame = cv2.resize(annotated_frame, (0, 0), fx=scale, fy=scale)

        # 2. 顯示縮小後的畫面
        cv2.imshow("Video Inference (Press 'q' to exit)", resized_frame)

        # 按下鍵盤的 'q' 鍵可以提早結束播放
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("使用者中斷播放。")
            break

    # 釋放資源與關閉視窗
    cap.release()
    cv2.destroyAllWindows()
    logger.info("影片分析結束")


# ==========================================
# 主程式執行區
# ==========================================
if __name__ == "__main__":
    WEIGHT_FILE = "parrot_yolov8.pt"
    IMAGE_FILE = "napping_image.jpg"
    VIDEO_FILE = "eating_video.mp4"

    detect_image(WEIGHT_FILE, IMAGE_FILE)

    # 可在此處調整 scale 參數控制畫面大小：
    # scale=0.5 代表 50% 大小；若想更小可以設為 scale=0.3
    # detect_video(WEIGHT_FILE, VIDEO_FILE, scale=0.3)