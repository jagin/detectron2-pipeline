import cv2


def put_text(image, text, org, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5,
             color=(0, 0, 0), bg_color=None, thickness=1, line_type=cv2.LINE_AA,
             org_pos="tl", padding=2):
    x, y = org
    ret, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)

    # Calculate text and background box coordinates
    if org_pos == "tl":  # top-left origin
        bg_rect_pt1 = (x, y + ret[1] + baseline + 2 * padding)
        bg_rect_pt2 = (x + ret[0] + 2 * padding, y)
        text_org = (x + padding, y + ret[1] + padding)
    elif org_pos == "tr":  # top-right origin
        bg_rect_pt1 = (x - ret[0] - 2 * padding, y)
        bg_rect_pt2 = (x, y + ret[1] + baseline + 2 * padding)
        text_org = (x - ret[0] - padding, y + ret[1] + baseline + padding)
    elif org_pos == "bl":  # bottom-left origin
        bg_rect_pt1 = (x, y - ret[1] - baseline - 2 * padding)
        bg_rect_pt2 = (x + ret[0] + 2 * padding, y)
        text_org = (x + padding, y - baseline - padding)
    elif org_pos == "br":  # bottom-right origin
        bg_rect_pt1 = (x, y - ret[1] - baseline - 2 * padding)
        bg_rect_pt2 = (x - ret[0] - 2 * padding, y)
        text_org = (x - ret[0] - padding, y - baseline - padding)

    if bg_color:
        # Draw background box
        cv2.rectangle(image, bg_rect_pt1, bg_rect_pt2, bg_color, -1)

    cv2.putText(image,
                text=text,
                org=text_org,
                fontFace=font_face,
                fontScale=font_scale,
                color=color,
                thickness=thickness,
                lineType=line_type)
