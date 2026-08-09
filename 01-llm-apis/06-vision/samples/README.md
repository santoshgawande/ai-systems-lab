# Sample Images

Put your test images here. Gitignored (no binaries in the repo).

## Recommended test images to download

| File to save as | Good for testing |
|---|---|
| `photo.jpg` | General scene — a room, street, landscape |
| `screenshot.png` | UI screenshot — test OCR extraction |
| `chart.png` | Bar/line chart — test data extraction |
| `invoice.jpg` | Document with text — test structured OCR |
| `diagram.png` | Technical diagram — test description |

## Quick download (public domain images)

```bash
# Wikimedia Commons CC0 images
curl -Lo samples/photo.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png"
```

Or just copy any .jpg / .png from your machine into this folder.

## Supported formats

JPG, JPEG, PNG, GIF, WEBP — anything Pillow can open.
