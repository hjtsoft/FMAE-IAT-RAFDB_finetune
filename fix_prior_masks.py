import numpy as np

# 加载已生成的掩码
train_masks = np.load(
    '/Data/hjt/NLA/datasets/RAF-DB/basic/Annotation/au_prior/au_prior_train.npy',
    allow_pickle=True).item()
test_masks = np.load(
    '/Data/hjt/NLA/datasets/RAF-DB/basic/Annotation/au_prior/au_prior_test.npy',
    allow_pickle=True).item()

# 找出全1掩码（即检测失败的）
def is_fallback(mask):
    return mask.min() == 1.0 and mask.max() == 1.0

# 用成功样本的均值替换失败样本
success = [v for v in train_masks.values() if not is_fallback(v)]
mean_mask = np.stack(success).mean(axis=0)
print(f"成功样本数: {len(success)}，平均掩码均值: {mean_mask.mean():.4f}")

fixed_train, fixed_test = 0, 0
for k in train_masks:
    if is_fallback(train_masks[k]):
        train_masks[k] = mean_mask
        fixed_train += 1
for k in test_masks:
    if is_fallback(test_masks[k]):
        test_masks[k] = mean_mask
        fixed_test += 1

print(f"训练集替换: {fixed_train} 张，测试集替换: {fixed_test} 张")

# 覆盖保存
np.save('/Data/hjt/NLA/datasets/RAF-DB/basic/Annotation/au_prior/au_prior_train.npy', train_masks)
np.save('/Data/hjt/NLA/datasets/RAF-DB/basic/Annotation/au_prior/au_prior_test.npy', test_masks)
print("保存完成")