from flask import Flask, request, jsonify
import mysql.connector
import pandas as pd
import os
from Content_base_filtering_updated2 import CourseRecommender

app = Flask(__name__)

# Cấu hình kết nối database MySQL
DB_CONFIG = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'moodle',
    'raise_on_warnings': True
}

def get_connection():
    # Hàm tạo kết nối tới database
    return mysql.connector.connect(**DB_CONFIG)

def get_courses():
    # Lấy danh sách các khóa học từ database
    conn = get_connection()
    sql = """
        SELECT 
            c.id AS course_code,
            c.fullname AS course_name,
            c.summary AS course_description,
            cat.name AS course_category,
            c.semester
        FROM mdl_course c
        JOIN mdl_course_categories cat ON cat.id = c.category
        WHERE c.visible = 1 AND c.id > 1
    """
    courses = pd.read_sql(sql, conn)
    conn.close()
    # Chuẩn hóa kiểu dữ liệu cho course_code và semester
    courses["course_code"] = courses["course_code"].astype(str).str.strip().str.upper()
    if "semester" in courses.columns:
        courses["semester"] = courses["semester"].astype(str).str.strip()
    return courses

def get_courses_taken(userid):
    # Lấy danh sách mã các môn học mà sinh viên đã đăng ký
    conn = get_connection()
    cursor = conn.cursor()
    sql = """
        SELECT c.id AS course_code
        FROM mdl_course c
        JOIN mdl_enrol e ON e.courseid = c.id
        JOIN mdl_user_enrolments ue ON ue.enrolid = e.id
        WHERE ue.userid = %s
    """
    cursor.execute(sql, (userid,))
    result = [str(row[0]).strip().upper() for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return result

def get_prerequisite_map():
    # Lấy map các môn học và môn tiên quyết của chúng
    conn = get_connection()
    sql = "SELECT course_code, prerequisite_code FROM mdl_course_prerequisite"
    df = pd.read_sql(sql, conn)
    conn.close()
    preq_map = {}
    for row in df.itertuples(index=False):
        code = str(row.course_code).strip().upper()
        preq = str(row.prerequisite_code).strip().upper()
        if code not in preq_map:
            preq_map[code] = set()
        if preq:
            preq_map[code].add(preq)
    return preq_map

def get_missing_prereqs(course_code, courses_taken_set, prereq_map):
    # Trả về danh sách môn tiên quyết còn thiếu cho một môn học
    required = prereq_map.get(str(course_code).strip().upper(), set())
    return list(required - courses_taken_set) if required else []

def get_user_profile(userid):
    # Lấy thông tin hồ sơ người dùng
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT firstname, lastname, email, city, country FROM mdl_user WHERE id=%s", (userid,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user

def get_course_names_from_codes(course_codes, courses_df):
    # Chuyển mã môn học thành tên môn học
    course_names = []
    for code in course_codes:
        row = courses_df[courses_df['course_code'] == str(code).strip().upper()]
        if not row.empty:
            name = row['course_name'].values[0]
            course_names.append(name)
        else:
            course_names.append(f"[Môn không xác định: {code}]")
    return course_names

def get_passed_courses(userid):
    # Lấy danh sách mã các môn học đã qua (điểm >= 5)
    conn = get_connection()
    cursor = conn.cursor()
    sql = """
        SELECT c.id AS course_code
        FROM mdl_course c
        JOIN mdl_enrol e ON e.courseid = c.id
        JOIN mdl_user_enrolments ue ON ue.enrolid = e.id
        JOIN mdl_grade_items gi ON gi.courseid = c.id AND gi.itemtype = 'course'
        JOIN mdl_grade_grades gg ON gg.itemid = gi.id AND gg.userid = ue.userid
        WHERE ue.userid = %s AND gg.finalgrade IS NOT NULL AND gg.finalgrade >= 5
    """
    cursor.execute(sql, (userid,))
    result = [str(row[0]).strip().upper() for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return result

def get_failed_courses(userid):
    # Lấy danh sách mã các môn học chưa qua (điểm < 5)
    conn = get_connection()
    cursor = conn.cursor()
    sql = """
        SELECT c.id AS course_code
        FROM mdl_course c
        JOIN mdl_enrol e ON e.courseid = c.id
        JOIN mdl_user_enrolments ue ON ue.enrolid = e.id
        JOIN mdl_grade_items gi ON gi.courseid = c.id AND gi.itemtype = 'course'
        JOIN mdl_grade_grades gg ON gg.itemid = gi.id AND gg.userid = ue.userid
        WHERE ue.userid = %s AND gg.finalgrade IS NOT NULL AND gg.finalgrade < 5
    """
    cursor.execute(sql, (userid,))
    result = [str(row[0]).strip().upper() for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return result

@app.route('/recommend', methods=['POST'])
def recommend():
    # API endpoint nhận userid và trả về danh sách gợi ý môn học
    data = request.get_json()
    userid = data.get('userid')
    top_k = int(data.get('top_k', 5))

    if not userid:
        return jsonify({'error': 'Missing userid'}), 400

    user = get_user_profile(userid)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    courses_df = get_courses()
    if courses_df.empty:
        return jsonify({'error': 'No courses found'}), 500
    
    passed_courses = get_passed_courses(userid)
    failed_courses = get_failed_courses(userid)
    courses_taken = get_courses_taken(userid)
    courses_taken_set = set(str(x).strip().upper() for x in courses_taken)

    # Xác định học kỳ hiện tại dựa trên các môn đã học
    taken_df = courses_df[courses_df["course_code"].isin(courses_taken_set)]
    semesters_taken = (
        taken_df["semester"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(float)
        .dropna()
        .astype(int)
    )
    current_semester = semesters_taken.max() if not semesters_taken.empty else 1

    # Lấy tên các môn đã qua và chưa qua
    passed_names = get_course_names_from_codes(passed_courses, courses_df)
    failed_names = get_course_names_from_codes(failed_courses, courses_df)
    
    learned_text = (
        f"Đã hoàn tất các môn: {', '.join(passed_names)}."
        if passed_names else "Chưa hoàn tất môn học nào."
    )

    retake_text = (
        f"Cần học lại các môn: {', '.join(failed_names)}."
        if failed_names else "Không có môn nào cần học lại."
    )

    # Tạo profile text cho sinh viên
    profile_text = f"""
    {user['firstname']} {user['lastname']} sống tại {user['city']}, quốc gia {user['country']}, email {user['email']}.
    Sinh viên hiện đang học kỳ {current_semester}.
    {learned_text}
    {retake_text}
    """

    prereq_map = get_prerequisite_map()

    # Khởi tạo và huấn luyện thuật toán gợi ý
    recommender = CourseRecommender(model_name="output_finetuned_vn") # Đảm bảo model_name trỏ đến mô hình đã huấn luyện
    recommender.fit(courses_df)

    # Tạo vector embedding từ hồ sơ sinh viên
    emb = recommender.embed_text(profile_text)

    # Gợi ý các môn học phù hợp (loại trừ các môn đã qua)
    recs_raw = recommender.recommend(
        emb,
        exclude_codes=passed_courses,
        top_k=top_k
    )

    recs = []
    for code, score in recs_raw:
        row = courses_df.loc[courses_df["course_code"] == code]
        cname = row["course_name"].values[0] if not row.empty else "Unknown"
        cdesc = row["course_description"].values[0] if not row.empty else ""
        semester_label = row["semester"].values[0] if not row.empty and "semester" in row.columns else ""

        prereq = list(prereq_map.get(str(code).strip().upper(), set()))
        missing_prereq = get_missing_prereqs(code, courses_taken_set, prereq_map)
        missing_prereq_names = get_course_names_from_codes(missing_prereq, courses_df)

        recs.append({
            'course_code': code,
            'course_name': cname,
            'score': float(score),
            'course_description': cdesc,
            'semester': f"HK{semester_label:02d}" if isinstance(semester_label, int) else semester_label,
            'prerequisite': prereq,
            'missing_prerequisite': missing_prereq_names,
            'roadmap_url': f'/course/view.php?id={code}'
        })

    return jsonify(recs)

if __name__ == '__main__':
    # Chạy Flask app ở chế độ debug
    app.run(host='0.0.0.0', port=8890, debug=True)
