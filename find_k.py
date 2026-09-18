import math
import numpy as np

def predict_knn(dataset, x_new, k):
    # ตรวจสอบความถูกต้อง
    if len(dataset) == 0:
        raise ValueError("Dataset ต้องไม่เป็นค่าว่าง")
    
    # จุดอ้างอิง
    expected_dim = len(dataset[0][0])
    
    for index, item in enumerate(dataset):
        # ตรวจสอบรูปแบบ []
        if len(item) != 2:
            raise ValueError(f"Error: จุดที่ {index+1} ต้องมีโครงสร้างเป็น [พิกัด, 'ชื่อกลุ่ม']")
        
        point = item[0]
        label = item[1]
        
        # ตรวจสอบว่าชื่อกลุ่ม
        if not isinstance(label, str) or label.strip() == "":
            raise ValueError(f"Error: จุดที่ {index+1} ต้องมีชื่อกลุ่มแนบมาด้วยเสมอ")
            
        # ตรวจสอบมิติเท่ากันไหม
        if len(point) != expected_dim:
            raise ValueError(f"Error: จุดที่ {index+1} มีมิติไม่เท่ากับจุดอื่น (ต้องมี {expected_dim} มิติ)")
    
    # ตรวจสอบมิติเท่ากันไหม (ของอ้างอิง)
    if len(x_new) != expected_dim:
        raise ValueError(f"Error: จุดพยากรณ์ต้องมี {expected_dim} มิติเท่ากับ Dataset")

    # คำนวณระยะห่าง
    distances = []
    x_new_array = np.array(x_new, dtype=float)
    
    for item in dataset:
        point = item[0]
        label = item[1]
        
        point_array = np.array(point, dtype=float)
        
        dist = math.dist(point_array, x_new_array)
        distances.append((dist, label))

    # น้อยไปมาก
    distances.sort(key=lambda x: x[0])

    # เลือกค่า K
    k_nearest_labels = []
    for i in range(k):
        label = distances[i][1]
        k_nearest_labels.append(label)

    # Majority Vote
    unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
    max_count_index = np.argmax(counts)
    prediction = unique_labels[max_count_index]

    return prediction

if __name__ == "__main__":

    # Input พิกัด
    input_1_dataset = [
        [[1.0, 2.0, 1.5], "Group_A"],
        [[2.0, 3.0, 2.1], "Group_A"],
        [[5.0, 6.0, 5.5], "Group_B"],
        [[6.0, 7.0, 6.2], "Group_B"]
    ]

    # จุดที่ต้องการพยากรณ์
    input_2_target = [1.5, 2.5, 1.8]

    # ค่า k
    input_3_k = 3

    # เรียก function
    result = predict_knn(input_1_dataset, input_2_target, input_3_k)
    print("ผลการพยากรณ์กลุ่มคือ:", result)