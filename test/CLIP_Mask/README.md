# ..

Dataset dir:
```json
dataset/
├── images/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
├── annotations.json

```

Dataset: 
```json
[
  {
    "image_id": "0001.jpg",
    "text_prompt": "the red mug on the table",
    "objects": [
      {
        "id": 1,
        "box": [x, y, width, height],
        "mask_path": "masks/0001_obj1.png",  // 二值掩膜图像
        "label": "red mug"
      }
    ]
  },
  {
    "image_id": "0002.jpg",
    "text_prompt": "pick up the small blue book",
    "objects": [
      {
        "id": 1,
        "box": [x, y, width, height],
        "mask_path": "masks/0002_obj1.png",
        "label": "blue book"
      }
    ]
  }
]

```