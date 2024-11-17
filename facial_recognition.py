import cv2
from deepface import DeepFace
from assembleModel import predict
#import mariadb
import shutil


def verify_face(img_path):
    img = cv2.imread(img_path)
    '''
    try:
        face_objs = DeepFace.extract_faces(img_path, detector_backend='opencv')
    except ValueError:
        return (False, 'Cant face detection')

    if len(face_objs) != 1:
        return (False, 'Cant face detection')

    facial_area = face_objs[0]['facial_area']
    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

    # 패딩 비율만큼 여유 공간을 추가
    pad_x = int(w * 0.35)
    pad_y = int(h * 0.35)

    # 확장된 얼굴 영역 계산 (이미지 경계를 넘지 않도록 제한)
    start_x = max(0, x - pad_x)
    start_y = max(0, y - pad_y)
    end_x = min(img.shape[1], x + w + pad_x)
    end_y = min(img.shape[0], y + h + pad_y)

    # 더 넓게 자른 얼굴 이미지 추출
    wider_face = img[start_y:end_y, start_x:end_x]
    '''
    result = predict(img)

    if result == 'Deepfake':
        return (False, 'Deepfake')
    else:
        return (True, 'Real')


def find_face(img_path):

    img = cv2.imread(img_path)
    # 얼굴 이미지 저장 경로
    db_path = './Member'
    path = None

    '''
    verified, resultText = verify_face(img_path)
    if verified is False:
        return (False, resultText)
    '''

    result = verify_face(img_path)
    if result[0] is False:
        return result

    try:
        result = DeepFace.find(img_path=img,
                               db_path=db_path,
                               detector_backend='opencv',
                               model_name='ArcFace')
        path = result[0]['identity'][0].replace("\\", "/")
        distance = result[0]['distance'][0]

    except ValueError:
        print("얼굴을 찾을 수 없습니다.")
        return (False, 'Cant face process')
    except Exception as e:
        print(e)
        return (False, 'Cant face process')
    '''
    try:
        conn = mariadb.connect(
            user="root",
            password="1234",
            host="localhost",
            port=3306,
            database="face"
        )
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        return (False, 'MariaDB cant connect')

    # Get Cursor
    cur = conn.cursor()

    insert_query = f"SELECT name FROM member WHERE img_path='{path}'"

    try:
        cur.execute(insert_query)
        query_result = cur.fetchall()[0][0]
    except mariadb.Error as e:
        print(f"Error: {e}")
        conn.close()
        return (False, 'No Result')
    '''

    if distance < 0.52:
        #print(query_result)
        #return (True, query_result)
        print("good")
        return (True, "KimHongJun")
    else:
        return (False, "No Matching")

def register_face(img_paths, name, birthday):

    img_file_list = list()
    for i in range(0, len(img_paths)):
        img_path = img_paths[i]
        img = cv2.imread(img_path)

        destination = './Member/' + name + '_' + birthday + '_' + str(i) + '.jpg'
        img_file_list.append(destination)
        shutil.copyfile(img_path, destination)

    '''
    # Connect to MariaDB Platform
    try:
        conn = mariadb.connect(
            user="xxxx",
            password="xxxx",
            host="localhost",
            port=0000,
            database="res"
        )
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        return (False, 'MariaDB cant connect')

    # Get Cursor
    cur = conn.cursor()

    insert_query = "INSERT INTO member (name, birth, img_path) VALUES (?, ?, ?)"

    try:
        cur.execute(insert_query, (name, birthday, destination))
    except mariadb.Error as e:
        print(f"Error: {e}")
        conn.close()
        return (False, 'already entry')

    conn.commit()
    conn.close()
    '''
    return (True, 'success')


if __name__ == '__main__':
    verified, result = verify_face(f'./test.jpg')
    print(result)
    #find_face('./Left/Karina2.jpg')
    #register_face('./uploads', 'Karina1.jpg', 'Karina', '2000-01-01')


# 얼굴 검출 모델 목록 (원하는 모델 선택 사용) (retinaface가 가장 느림)
#detection_models = ['opencv', 'ssd', 'mtcnn', 'retinaface']

# 얼굴 표현 모델 목록 (원하는 모델 선택 사용)
#embedding_models = ['VGG-Face', 'Facenet', 'Facenet512', 'DeepID', 'ArcFace', 'SFace']

