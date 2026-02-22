"""Prepare Vietnamese product review data for sentiment classification.

Downloads and preprocesses Vietnamese review datasets for 3-class
sentiment classification (positive, negative, neutral).

Supports:
- UIT-VSFC (Vietnamese Students' Feedback Corpus)
- Synthetic Vietnamese product reviews for quick testing
"""

import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.dirname(__file__)

# Sentiment labels
LABELS = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}


def generate_synthetic_reviews(n_samples=10000, seed=42):
    """Generate synthetic Vietnamese product reviews for training.

    Creates realistic-looking Vietnamese review text with sentiment labels.
    Used for development/testing when real datasets aren't available.

    Args:
        n_samples: Total number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with 'text' and 'label' columns.
    """
    random.seed(seed)
    np.random.seed(seed)

    positive_templates = [
        "Sản phẩm rất tốt, tôi rất hài lòng với chất lượng",
        "Giao hàng nhanh, đóng gói cẩn thận, hàng đẹp lắm",
        "Chất lượng tuyệt vời, giá cả hợp lý, sẽ mua lại",
        "Sản phẩm đúng mô tả, chất liệu tốt, rất đáng tiền",
        "Tôi rất thích sản phẩm này, dùng rất tốt",
        "Hàng chính hãng, chất lượng cao, shop uy tín",
        "Sản phẩm vượt ngoài mong đợi, rất tuyệt vời",
        "Mua lần thứ 3 rồi, luôn hài lòng với chất lượng",
        "Sản phẩm xịn, giao hàng nhanh, đánh giá 5 sao",
        "Chất lượng rất ok, giá hợp lý, sẽ giới thiệu cho bạn bè",
        "Hàng đẹp, giá tốt, shop phục vụ nhiệt tình",
        "Sản phẩm dùng rất thích, chất liệu cao cấp",
        "Rất hài lòng, sản phẩm giống hình, giao hàng nhanh",
        "Tuyệt vời luôn, đúng như mong đợi, cảm ơn shop",
        "Sản phẩm chất lượng, đóng gói đẹp, sẽ ủng hộ tiếp",
    ]

    negative_templates = [
        "Sản phẩm kém chất lượng, không giống mô tả",
        "Hàng bị lỗi, giao hàng chậm, rất thất vọng",
        "Chất lượng tệ, không đáng tiền, không nên mua",
        "Sản phẩm khác xa so với hình ảnh, tôi rất buồn",
        "Giao hàng quá chậm, đóng gói sơ sài, hàng bị hỏng",
        "Sản phẩm không dùng được, chất liệu rất tệ",
        "Mua về dùng được 2 ngày đã hỏng, rất thất vọng",
        "Hàng fake, không phải hàng chính hãng như quảng cáo",
        "Shop giao sai hàng, liên hệ không được, tệ quá",
        "Sản phẩm rất kém, không đáng mua, lãng phí tiền",
        "Hàng nhận được bị vỡ, đóng gói quá cẩu thả",
        "Chất lượng không như mong đợi, giá thì đắt",
        "Sản phẩm dở, không giống quảng cáo chút nào",
        "Rất tệ, sản phẩm hỏng ngay khi mở hộp",
        "Không hài lòng chút nào, sẽ không bao giờ mua lại",
    ]

    neutral_templates = [
        "Sản phẩm bình thường, không có gì đặc biệt",
        "Hàng tạm ổn, giao hàng bình thường",
        "Sản phẩm được, giá hợp lý, chất lượng tạm",
        "Nhận hàng rồi, chưa dùng nên chưa đánh giá được",
        "Sản phẩm OK, không tốt không xấu",
        "Hàng đúng mô tả, giá tầm trung, chất lượng ổn",
        "Giao hàng hơi chậm nhưng sản phẩm cũng được",
        "Sản phẩm tạm được, cần dùng thêm mới đánh giá",
        "Hàng bình thường, đúng giá tiền",
        "Shop giao hàng đúng hẹn, sản phẩm tạm ổn",
        "Chất lượng trung bình, không quá xuất sắc",
        "Sản phẩm ổn trong tầm giá, giao hàng bình thường",
        "Mới nhận hàng, nhìn bên ngoài cũng ổn",
        "Hàng giống mô tả, chất lượng trung bình",
        "Sản phẩm dùng được, giá cả phải chăng",
    ]

    # Product terms to inject for variety
    products = [
        "áo thun", "quần jeans", "giày dép", "túi xách", "đồng hồ",
        "tai nghe", "ốp điện thoại", "bàn phím", "chuột máy tính", "balo",
        "mỹ phẩm", "kem dưỡng", "son môi", "nước hoa", "sách",
        "đèn bàn", "quạt mini", "bình giữ nhiệt", "kính mát", "nón",
    ]

    adjectives_pos = ["đẹp", "tốt", "chất lượng", "xịn", "bền", "mềm mại"]
    adjectives_neg = ["tệ", "xấu", "kém", "mỏng", "dở", "rách"]
    adjectives_neu = ["bình thường", "tạm", "ổn", "được", "vậy thôi"]

    data = []
    label_counts = {0: 0, 1: 0, 2: 0}
    samples_per_class = n_samples // 3

    for label, templates, adjs in [
        (2, positive_templates, adjectives_pos),
        (0, negative_templates, adjectives_neg),
        (1, neutral_templates, adjectives_neu),
    ]:
        for i in range(samples_per_class):
            template = random.choice(templates)
            product = random.choice(products)
            adj = random.choice(adjs)

            # Add some variation
            if random.random() > 0.5:
                text = f"{template}. {product.capitalize()} {adj}."
            elif random.random() > 0.5:
                text = f"Mua {product} về dùng. {template}"
            else:
                text = template

            data.append({"text": text, "label": label})
            label_counts[label] += 1

    # Fill remaining
    while len(data) < n_samples:
        label = random.choice([0, 1, 2])
        templates = [positive_templates, neutral_templates, negative_templates][label if label != 0 else 2]
        data.append({"text": random.choice(templates), "label": label})

    random.shuffle(data)
    df = pd.DataFrame(data)

    print(f"Generated {len(df)} samples")
    print(f"Label distribution: {df['label'].value_counts().sort_index().to_dict()}")

    return df


def save_splits(df, output_dir=DATA_DIR, test_size=0.15, val_size=0.15):
    """Split data and save as CSV files.

    Args:
        df: DataFrame with 'text' and 'label' columns.
        output_dir: Where to save CSVs.
        test_size: Test set ratio.
        val_size: Validation set ratio (from remaining after test).
    """
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label"]
    )
    train, val = train_test_split(
        train_val, test_size=val_size / (1 - test_size),
        random_state=42, stratify=train_val["label"]
    )

    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"\nSaved splits:")
    print(f"  Train: {len(train)} samples")
    print(f"  Val:   {len(val)} samples")
    print(f"  Test:  {len(test)} samples")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Vietnamese sentiment data")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--source", choices=["synthetic"], default="synthetic")
    args = parser.parse_args()

    df = generate_synthetic_reviews(n_samples=args.n_samples)
    save_splits(df)
