## Evaluation

### 1. Panorama datasets
<table style="width:100%">
  <tr>
    <th>Dataset details</th>
    <th>Count</th>
  </tr>
  <tr>
    <td><b>Total Individuals</b></td>
    <td><b>3</b></td>
  </tr>
  <tr>
    <td>Images per Individual</td>
    <td>5</td>
  </tr>
  <tr>
    <td><b>Total Images</b></td>
    <td><b>15</b></td>
  </tr>
</table>
  
### 2. Performace evaluation
The performance evaluation was conducted using the following five metrics: L1, Cosine Distance, LPIPS, SSIM, and PSNR.
#### 2.1 Comparison
<table style="width:100%">
  <tr>
    <th>Method</th>
    <th>L1↓</th>
    <th>Cos Dist.↓</th>
    <th>LPIPS↓</th>
    <th>SSIM↑</th>
    <th>PSNR↑</th>
  </tr>
  <tr>
    <td>IPA</td>
    <td>1.099</td>
    <td>0.714</td>
    <td>0.767</td>
    <td>0.403</td>
    <td>10.454</td>
  </tr>
  <tr>
    <td>IPA-Plus</td>
    <td>1.129</td>
    <td>0.752</td>
    <td>0.740</td>
    <td>0.445</td>
    <td>11.626</td>
  </tr>
  <tr>
    <td><b>IPA-Plus-FaceID</b></td>
    <td><b>1.074</b></td>
    <td><b>0.678</b></td>
    <td><b>0.676</b></td>
    <td><b>0.492</b></td>
    <td><b>12.758</b></td>
  </tr>
</table>
<table style="width:100%">
  <tr>
    <th>Method</th>
    <th>L1↓</th>
    <th>Cos Dist.↓</th>
    <th>LPIPS↓</th>
    <th>SSIM↑</th>
    <th>PSNR↑</th>
  </tr>
  <tr>
    <td>LoRA</td>
    <td>1.169</td>
    <td>0.845</td>
    <td>0.750</td>
    <td>0.426</td>
    <td>11.106</td>
  </tr>
  <tr>
    <td>InstantID</td>
    <td>1.064</td>
    <td>0.694</td>
    <td>0.762</td>
    <td>0.368</td>
    <td>10.666</td>
  </tr>
  <tr>
    <td><b>TIFF (Ours)</b></td>
    <td><b>0.892</b></td>
    <td><b>0.478</b></td>
    <td><b>0.691</b></td>
    <td><b>0.431</b></td>
    <td><b>12.406</b></td>
  </tr>
  <tr>
    <td>Ref Image (GT)</td>
    <td>0.838</td>
    <td>0.419</td>
    <td>0.386</td>
    <td>0.546</td>
    <td>inf.</td>
  </tr>
</table>
<table style="width:100%">
  <tr>
    <th>Method</th>
    <th>L1↓</th>
    <th>Cos Dist.↓</th>
    <th>LPIPS↓</th>
    <th>SSIM↑</th>
    <th>PSNR↑</th>
  </tr>
  <tr>
    <td>LoRA</td>
    <td>1.251</td>
    <td>0.969</td>
    <td>0.773</td>
    <td>0.434</td>
    <td>10.424</td>
  </tr>
  <tr>
    <td>InstantID</td>
    <td>1.222</td>
    <td>0.924</td>
    <td>0.785</td>
    <td>0.397</td>
    <td>10.869</td>
  </tr>
  <tr>
    <td><b>TIFF (Ours)</b></td>
    <td><b>0.967</b></td>
    <td><b>0.568</b></td>
    <td><b>0.723</b></td>
    <td><b>0.438</b></td>
    <td><b>12.293</b></td>
  </tr>
  <tr>
    <td>Ref Video (GT)</td>
    <td>0.950</td>
    <td>0.519</td>
    <td>0.605</td>
    <td>0.442</td>
    <td>11.315</td>
  </tr>
</table>
#### 2.2 Ablation Studies
<table style="width:100%">
  <tr>
    <th>Method</th>
    <th>L1↓</th>
    <th>Cos Dist.↓</th>
    <th>LPIPS↓</th>
    <th>SSIM↑</th>
    <th>PSNR↑</th>
  </tr>
  <tr>
    <td>w/o IPA</td>
    <td>1.116</td>
    <td>0.745</td>
    <td>0.751</td>
    <td>0.420</td>
    <td>10.445</td>
  </tr>
  <tr>
    <td><b>with IPA</b></td>
    <td><b>1.076</b></td>
    <td><b>0.686</b></td>
    <td><b>0.716</b></td>
    <td><b>0.461</b></td>
    <td><b>11.968</b></td>
  </tr>
</table>
<table style="width:100%">
  <tr>
    <th>Method</th>
    <th>L1↓</th>
    <th>Cos Dist.↓</th>
    <th>LPIPS↓</th>
    <th>SSIM↑</th>
    <th>PSNR↑</th>
  </tr>
  <tr>
    <td>w/o Face Reconstruction</td>
    <td>0.821</td>
    <td>0.394</td>
    <td>0.697</td>
    <td>0.499</td>
    <td>12.686</td>
  </tr>
  <tr>
    <td>Code Former</td>
    <td>0.829</td>
    <td>0.400</td>
    <td>0.687</td>
    <td>0.490</td>
    <td>12.626</td>
  </tr>
  <tr>
    <td><b>GFPGAN v1.4</b></td>
    <td><b>0.814</b></td>
    <td><b>0.385</b></td>
    <td><b>0.692</b></td>
    <td><b>0.496</b></td>
    <td><b>12.682</b></td>
  </tr>
</table>
<table style="width:100%">
  <tr>
    <th>Method</th>
    <th>L1↓</th>
    <th>Cos Dist.↓</th>
    <th>LPIPS↓</th>
    <th>SSIM↑</th>
    <th>PSNR↑</th>
  </tr>
  <tr>
    <td>w/o Face Module</td>
    <td>1.168</td>
    <td>0.844</td>
    <td>0.773</td>
    <td>0.388</td>
    <td>11.191</td>
  </tr>
  <tr>
    <td><b>With Face Module</b></td>
    <td><b>0.949</b></td>
    <td><b>0.545</b></td>
    <td><b>0.751</b></td>
    <td><b>0.402</b></td>
    <td><b>11.601</b></td>
  </tr>
</table>
<table style="width:100%">
  <tr>
    <th>Method</th>
    <th>L1↓</th>
    <th>Cos Dist.↓</th>
    <th>LPIPS↓</th>
    <th>SSIM↑</th>
    <th>PSNR↑</th>
  </tr>
  <tr>
    <td>w/o IPA Face Module</td>
    <td>1.131</td>
    <td>0.794</td>
    <td>0.772</td>
    <td>0.383</td>
    <td>11.097</td>
  </tr>
  <tr>
    <td><b>With IPA Face Module</b></td>
    <td><b>0.987</b></td>
    <td><b>0.596</b></td>
    <td><b>0.752</b></td>
    <td><b>0.407</b></td>
    <td><b>11.694</b></td>
  </tr>
</table>

### 3. Validation

```shell
# evaluation
python3 eval.py \
--gt_images_path ./data/gt/ \                  # Path to the ground truth images
--validation_images_path ./data/validation/ \  # Path to the validation images
--result_path ./results/                       # Path to save the evaluation csv results
```
