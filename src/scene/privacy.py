"""
privacy.py — face blur enforcement
Applied BEFORE any egress (storage, network, UI).
Only the face bounding box is blurred; chest/abdomen untouched.
"""
# TODO: implement — MediaPipe FaceDetection -> Gaussian blur on bbox
