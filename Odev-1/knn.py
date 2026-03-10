import numpy as np
import os
from PIL import Image

def load_data(folder):
    X = []
    y = []
    class_names = sorted(os.listdir(folder))  # sınıf isimlerini alfabetik sırayla al
    class_to_idx = {name:i for i,name in enumerate(class_names)}
    
    for class_name in class_names:
        class_folder = os.path.join(folder, class_name)
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            img = Image.open(img_path).resize((32,32))   # 32x32 boyut
            img_array = np.array(img).flatten()          # 1D vektör
            X.append(img_array)
            y.append(class_to_idx[class_name])
    
    return np.array(X), np.array(y), class_to_idx

# Veriyi yükle
X_train, y_train, class_to_idx = load_data(os.path.join("data", "train"))
X_test, y_test, _ = load_data(os.path.join("data", "test"))

# Kullanıcıdan mesafe ve k
print("Mesafe seçin:")
print("1 - L1 (Manhattan)")
print("2 - L2 (Euclidean)")
secim = int(input("Seçim (1 veya 2): "))
k = int(input("k değeri girin: "))

num_test = 10
dogru = 0

for i in range(num_test):
    if secim == 1:
        distances = np.sum(np.abs(X_train - X_test[i]), axis=1)
    else:
        distances = np.sqrt(np.sum((X_train - X_test[i])**2, axis=1))
    
    nearest = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest]
    
    prediction = np.argmax(np.bincount(nearest_labels))
    
    # Tahmini ve gerçek sınıf ismini yaz
    inv_class_map = {v:k for k,v in class_to_idx.items()}
    print(f"{i+1}. Test Örneği -> Gerçek: {inv_class_map[y_test[i]]}, Tahmin: {inv_class_map[prediction]}")
    
    if prediction == y_test[i]:
        dogru += 1

accuracy = dogru / num_test
print(f"\nToplam Doğruluk: {accuracy:.2f}")