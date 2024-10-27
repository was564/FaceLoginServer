import cv2
from deepface import DeepFace
#import mariadb
import shutil


def verify_face(img_path):
    try:
        face_objs = DeepFace.extract_faces(img_path, anti_spoofing=True)
    except ValueError:
        return (False, 'Cant face detection')

    face_count = 0
    for face_obj in face_objs:
        score = face_obj['confidence']
        if score > 0.90:
            face_count += 1
            face = face_obj

    if face_count != 1:
        print(face_count)
        return (False, 'Cant face detection')
    #elif face["is_real"] is False:
    #    return (False, 'Face like fake')
    else:
        return (True, 'verify Face')


def find_face(img_path):

    img = cv2.imread(img_path)
    # 얼굴 이미지 저장 경로
    db_path = './Member'
    path = None

    verified, resultText = verify_face(img_path)
    if verified is False:
        return (False, resultText)


    try:
        result = DeepFace.find(img_path=img,
                               db_path=db_path,
                               detector_backend='mtcnn',
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
        return (True, "Good")
    else:
        return (False, "No Matching")

def register_face(img_path, img_name, name, birthday):
    img_complete_path = img_path + '/' + img_name
    img = cv2.imread(img_complete_path)

    verified, resultText = verify_face(img)
    if verified is False:
        return (False, resultText)

    destination = './Member/' + img_name
    shutil.copyfile(img_complete_path, destination)

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
    return (True, 'success')


if __name__ == '__main__':
    verified, result = verify_face(f'./Left/test.jpg')
    print(result)
    #find_face('./Left/Karina2.jpg')
    #register_face('./uploads', 'Karina1.jpg', 'Karina', '2000-01-01')


# 얼굴 검출 모델 목록 (원하는 모델 선택 사용) (retinaface가 가장 느림)
#detection_models = ['opencv', 'ssd', 'mtcnn', 'retinaface']

# 얼굴 표현 모델 목록 (원하는 모델 선택 사용)
#embedding_models = ['VGG-Face', 'Facenet', 'Facenet512', 'DeepID', 'ArcFace', 'SFace']

