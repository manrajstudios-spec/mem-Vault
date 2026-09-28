from pdf2image import convert_from_path

doc_images = convert_from_path(
    "path",
    poppler_path="/usr/bin"
)