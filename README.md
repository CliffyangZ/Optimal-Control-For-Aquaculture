# Aquaculture_CV_Project

## 1. System Architecture
![System Architecture](assets/Sys_Arch.drawio.svg)

## 2. Software Architecture
Aquaculture_CV_Project/
├── assets/                  # 新增：存放靜態資源、Icon、或測試短片
├── config/
│   └── settings.py
├── streaming/
│   └── 
├── core/
│   ├── vision_engine.py
│   ├── logic_handler.py
│   
├── drivers/
│   ├── camera_stream.py
│   └── actuator.py
├── web/
│   ├── app.py
│   └── components/          # 新增：拆分 Streamlit 的側邊欄、圖表等組件
├── data/                    # 新增：存放 Log 與辨識截圖
│   └── captures/
├── main.py                  # 若採方案 A，此檔案可作為啟動檢查腳本

## 3. Development Principles

- Use Spec-Driven Development: write the English specification before planning or coding.
- Keep specifications, plans, tasks, code comments, and docstrings in English.
- Keep modules separated by responsibility: configuration, core logic, drivers, UI, and scripts.
- Prefer simple, direct implementations over speculative abstractions.
- Define verification steps before completing changes that affect vision, feeding, actuators, or safety.

## 4. Hardware Components

### The Brain
- MPU (Rasperry Pi Zero 2W)
- MCU (BMduino)
- FC (PixHawk)

### Drone Comopnents
- Camera (Rasperry Pi 5 IMX219 79.3 degree)
- Drone Basis (F450)
