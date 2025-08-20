import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image

class PathLossCalculator:
    """Path Loss Calculator"""
    
    def __init__(self):
        self.FREQ_DICT = {
            'f1': 868e6,   # 868 MHz
            'f2': 1.8e9,   # 1.8 GHz
            'f3': 3.5e9    # 3.5 GHz
        }
        self.PIXEL_SIZE = 0.25  # Spatial resolution (m)
        
    def load_antenna_pattern(self, antenna_id):
        """Load antenna radiation pattern"""
        pattern_file = f'datasets/ICASSP2025_Dataset/Radiation_Patterns/Ant{antenna_id}_Pattern.csv'
        pattern_data = pd.read_csv(pattern_file, header=None)
        return pattern_data.values.flatten()

    def load_tx_position(self, building_id, antenna_id, freq_id, sample_id):
        """Load transmitter position information"""
        pos_file = f'datasets/ICASSP2025_Dataset/Positions/Positions_B{building_id}_Ant{antenna_id}_f{freq_id}.csv'
        positions = pd.read_csv(pos_file)
        pos = positions.iloc[sample_id]
        return pos

    def load_input_image(self, building_id, antenna_id, freq_id, sample_id):
        """Load input image"""
        image_name = f'B{building_id}_Ant{antenna_id}_f{freq_id}_S{sample_id}'
        image_path = f'datasets/ICASSP2025_Dataset/Inputs/Task_3_ICASSP/{image_name}.png'
        return np.array(Image.open(image_path))

    def calculate_path_loss(self, power_levels, frequency_hz, grid_coords, tx_pos, tx_azimuth):
        """
        Calculate free space path loss considering antenna radiation pattern
        """
        c = 3e8  # Speed of light
        wavelength = c / frequency_hz
        x, y = grid_coords
        tx_x, tx_y = tx_pos
        
        x_meters = x * self.PIXEL_SIZE
        y_meters = y * self.PIXEL_SIZE
        tx_x_meters = tx_x * self.PIXEL_SIZE
        tx_y_meters = tx_y * self.PIXEL_SIZE
        
        dx = x_meters - tx_x_meters
        dy = y_meters - tx_y_meters
        distances = np.sqrt(dx**2 + dy**2)
        distances = np.maximum(distances, 0.1)
        
        angles = (np.degrees(np.arctan2(dy, dx)) + tx_azimuth) % 360
        angle_indices = np.round(angles).astype(int) % 360
        
        fspl_basic = 20 * np.log10(4 * np.pi * distances / wavelength)
        
        gain_linear = 10**(power_levels/10)
        max_gain = np.max(gain_linear)
        gain_normalized = gain_linear / max_gain
        
        pattern_factors = gain_normalized[angle_indices]
        pattern_factors = np.maximum(pattern_factors, 1e-10)
        
        total_path_loss = fspl_basic - 10 * np.log10(pattern_factors)
        
        return total_path_loss

    def create_path_loss_map(self, building_id, antenna_id, freq_id, sample_id):
        """Create path loss map for a specific scenario"""
        power_levels = self.load_antenna_pattern(antenna_id)
        tx_info = self.load_tx_position(building_id, antenna_id, freq_id, sample_id)
        input_image = self.load_input_image(building_id, antenna_id, freq_id, sample_id)
        
        h, w = input_image.shape[:2]
        x = np.arange(w)
        y = np.arange(h)
        Y,X = np.meshgrid(x, y)
        
        frequency = self.FREQ_DICT[f'f{freq_id}']
        path_loss = self.calculate_path_loss(
            power_levels,
            frequency,
            (X, Y),
            (tx_info['X'], tx_info['Y']),
            (tx_info['Azimuth']-90)%360
        )
        
        if path_loss.shape != (h, w):
            print("Warning: path_loss shape does not match input image dimensions.")
        
        return path_loss, input_image

    def save_combined_image(self, path_loss, input_image, save_path, image_name):
        """Combine path loss with input channels and save as PNG"""
        combined_image = np.dstack((input_image, path_loss))
        save_file = os.path.join(save_path, f'{image_name}.png')
        Image.fromarray(combined_image.astype(np.uint8)).save(save_file)

    def load_output_image(self, building_id, antenna_id, freq_id, sample_id):
        """Load output image"""
        image_name = f'B{building_id}_Ant{antenna_id}_f{freq_id}_S{sample_id}'
        image_path = f'datasets/ICASSP2025_Dataset/Outputs/Task_3_ICASSP/{image_name}.png'
        return np.array(Image.open(image_path))

    def analyze_error(self, calculated_pl, ground_truth):
        """
        分析预测路径损耗与真实值之间的误差
        
        返回:
        dict: 包含各种误差指标
        """
        # 将ground truth转换为float类型
        ground_truth = ground_truth.astype(float)
        
        # 计算误差指标
        error = calculated_pl - ground_truth
        metrics = {
            'MSE': np.mean(error**2),
            'RMSE': np.sqrt(np.mean(error**2)),
            'MAE': np.mean(np.abs(error)),
            'Max_Error': np.max(np.abs(error)),
            'Min_Error': np.min(np.abs(error)),
            'Mean_Error': np.mean(error),
            'Std_Error': np.std(error)
        }
        return metrics
    
    def visualize(self, path_loss, input_image, building_id, antenna_id, freq_id, sample_id):
        """
        Visualize path loss, input image, and ground truth output for checking
        """
        output_image = self.load_output_image(building_id, antenna_id, freq_id, sample_id)
        
        # 分离输入图像的通道
        reflectance = input_image[:,:,0]
        transmittance = input_image[:,:,1]
        distance = input_image[:,:,2]
        
        # 确保path_loss与输出图像尺寸一致
        h, w = output_image.shape[:2]
        path_loss_resized = path_loss[:h, :w]
        
        # 从距离通道验证Tx位置
        tx_y, tx_x = np.unravel_index(np.argmin(distance), distance.shape)
        print(f"Tx position from distance channel: ({tx_x}, {tx_y})")
        
        # # 统一颜色范围（对路径损耗图）
        # vmin_pl = 0
        # vmax_pl = 160
        # 统一颜色范围（对路径损耗图）
        vmin_pl = min(np.min(output_image), np.min(path_loss_resized))
        vmax_pl = max(np.max(output_image), np.max(path_loss_resized))
        
        # 创建2x2布局
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 2, height_ratios=[4, 4, 1])
        
        # 1. 反射率通道
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(reflectance, cmap='viridis')
        plt.colorbar(im1, ax=ax1, label='Reflectance')
        ax1.plot(tx_x, tx_y, 'r*', markersize=15, label='Tx')
        ax1.set_title('Input: Reflectance Channel')
        ax1.axis('equal')
        ax1.legend()
        
        # 2. 距离通道
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(distance, cmap='viridis')
        plt.colorbar(im2, ax=ax2, label='Distance')
        ax2.plot(tx_x, tx_y, 'r*', markersize=15, label='Tx')
        ax2.set_title('Input: Distance Channel')
        ax2.axis('equal')
        ax2.legend()
        
        # 3. 计算的路径损耗
        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.imshow(path_loss_resized, cmap='jet', vmin=vmin_pl, vmax=vmax_pl)
        plt.colorbar(im3, ax=ax3, label='Path Loss (dB)')
        ax3.plot(tx_x, tx_y, 'r*', markersize=15, label='Tx')
        ax3.set_title(f'Calculated Path Loss\nFreq: {self.FREQ_DICT[f"f{freq_id}"]/1e9:.1f} GHz')
        ax3.axis('equal')
        ax3.legend()
        
        # 4. 真实输出
        ax4 = fig.add_subplot(gs[1, 1])
        im4 = ax4.imshow(output_image, cmap='jet', vmin=vmin_pl, vmax=vmax_pl)
        plt.colorbar(im4, ax=ax4, label='Ground Truth Path Loss (dB)')
        ax4.plot(tx_x, tx_y, 'r*', markersize=15, label='Tx')
        ax4.set_title('Ground Truth')
        ax4.axis('equal')
        ax4.legend()
        
        # 5. 误差分析
        error_metrics = self.analyze_error(path_loss_resized, output_image)
        error_text = (
            f"Error Metrics:\n"
            f"RMSE: {error_metrics['RMSE']:.2f} dB\n"
            f"MAE: {error_metrics['MAE']:.2f} dB\n"
            f"Mean Error: {error_metrics['Mean_Error']:.2f} dB\n"
            f"Std Error: {error_metrics['Std_Error']:.2f} dB\n"
            f"Tx Position (x,y): ({tx_x}, {tx_y})"
        )
        print(error_text)
        
        # 设置总标题
        plt.suptitle(
            f'Path Loss Comparison\n'
            f'Building {building_id}, Antenna {antenna_id}, '
            f'Freq {self.FREQ_DICT[f"f{freq_id}"]/1e9:.1f} GHz, '
            f'Sample {sample_id}',
            fontsize=14
        )
        
        plt.tight_layout()
        plt.show()

    # def visualize(self, path_loss, input_image, building_id, antenna_id, freq_id, sample_id):
    #     """
    #     Visualize path loss, input image, and ground truth output for checking
    #     """
    #     output_image = self.load_output_image(building_id, antenna_id, freq_id, sample_id)
        
    #     plt.figure(figsize=(15, 5))
        
    #     # Plot input image channels
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(input_image[:,:,0])
    #     plt.title('Input Image')
    #     plt.axis('off')
        
    #     # Plot calculated path loss
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(path_loss, cmap='jet', vmin=0, vmax=160)
    #     plt.title('Calculated Path Loss')
    #     plt.axis('off')
        
    #     # Plot ground truth output
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(output_image, cmap='jet', vmin=0, vmax=160)
    #     plt.title('Ground Truth')
    #     plt.axis('off')
        
    #     plt.suptitle(f'B{building_id}_Ant{antenna_id}_f{freq_id}_S{sample_id}')
    #     plt.tight_layout()
    #     plt.show()

# def main():
#     calculator = PathLossCalculator()
    
#     input_dir = '/home/work/test_data/ICASSP2025_Dataset/Inputs/Task_3_ICASSP'
#     save_dir = 'path_loss_results'
#     os.makedirs(save_dir, exist_ok=True)
#     ii = 0
#     for file_name in os.listdir(input_dir):
#         ii += 1
#         if file_name.endswith('.png'):
#             parts = file_name.split('_')
#             building_id = int(parts[0][1:])
#             antenna_id = int(parts[1][3:])
#             freq_id = int(parts[2][1:])
#             sample_id = int(parts[3][1:].split('.')[0])
            
#             path_loss, input_image = calculator.create_path_loss_map(
#                 building_id, antenna_id, freq_id, sample_id
#             )
            
#             # Visualize for checking
#             calculator.visualize(path_loss, input_image, 
#                               building_id, antenna_id, freq_id, sample_id)
            
#             image_name = f'B{building_id}_Ant{antenna_id}_f{freq_id}_S{sample_id}'
#             calculator.save_combined_image(path_loss, input_image, save_dir, image_name)
#         if ii > 10:
#             break

# if __name__ == "__main__":
#     main()

    def visualize_combined_image(self, image_path):
        """
        Visualize the saved combined image with all channels
        
        Args:
            image_path: Path to the combined image (*_combined.png)
        """
        # 读取combined图像
        combined_image = np.array(Image.open(image_path))
        
        # 分离通道
        reflectance = combined_image[:,:,0]    # 反射率
        transmittance = combined_image[:,:,1]   # 透射率
        distance = combined_image[:,:,2]        # 距离
        path_loss = combined_image[:,:,3]       # 路径损耗
        
        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # 1. 反射率
        im0 = axes[0,0].imshow(reflectance, cmap='viridis')
        plt.colorbar(im0, ax=axes[0,0], label='Reflectance')
        axes[0,0].set_title('Channel 0: Reflectance')
        axes[0,0].axis('off')
        
        # 2. 透射率
        im1 = axes[0,1].imshow(transmittance, cmap='viridis')
        plt.colorbar(im1, ax=axes[0,1], label='Transmittance')
        axes[0,1].set_title('Channel 1: Transmittance')
        axes[0,1].axis('off')
        
        # 3. 距离
        im2 = axes[1,0].imshow(distance, cmap='viridis')
        plt.colorbar(im2, ax=axes[1,0], label='Distance')
        axes[1,0].set_title('Channel 2: Distance')
        axes[1,0].axis('off')
        
        # 4. 路径损耗
        im3 = axes[1,1].imshow(path_loss, cmap='jet', vmin=0, vmax=160)
        plt.colorbar(im3, ax=axes[1,1], label='Path Loss (dB)')
        axes[1,1].set_title('Channel 3: Path Loss')
        axes[1,1].axis('off')
        
        # 从文件名提取信息
        filename = os.path.basename(image_path)
        parts = filename.split('_')
        building_id = parts[0][1:]
        antenna_id = parts[1][3:]
        freq_id = parts[2][1:]
        sample_id = parts[3][1:].split('.')[0]
        
        plt.suptitle(
            f'Combined Image Channels\n'
            f'Building {building_id}, Antenna {antenna_id}, '
            f'Freq {freq_id}, Sample {sample_id}',
            fontsize=14
        )
        
        plt.tight_layout()
        plt.show()

# def main():
#     vis = True
#     calculator = PathLossCalculator()
    
#     # 原有的处理和保存逻辑
#     input_dir = '/home/work/test_data/ICASSP2025_Dataset/Inputs/Task_3_ICASSP'
#     save_dir = 'path_loss_results_minmax'
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 处理一些图片
#     ii = 0
#     for file_name in os.listdir(input_dir):
#         ii += 1
#         if file_name.endswith('.png'):
#             parts = file_name.split('_')
#             building_id = int(parts[0][1:])
#             antenna_id = int(parts[1][3:])
#             freq_id = int(parts[2][1:])
#             sample_id = int(parts[3][1:].split('.')[0])
            
#             path_loss, input_image = calculator.create_path_loss_map(
#                 building_id, antenna_id, freq_id, sample_id
#             )
#             if vis:
#                 # Visualize for checking
#                 calculator.visualize(path_loss, input_image, 
#                                 building_id, antenna_id, freq_id, sample_id)
                
#             image_name = f'B{building_id}_Ant{antenna_id}_f{freq_id}_S{sample_id}'
#             calculator.save_combined_image(path_loss, input_image, save_dir, image_name)
            
#             if vis:
#                 # 可视化保存的combined图片
#                 combined_path = os.path.join(save_dir, f'{image_name}.png')
#                 calculator.visualize_combined_image(combined_path)
#         if vis:    
#             if ii > 2:  # 只处理前两张图片作为示例
#                 break
#         # if ii > 2:  # 只处理前两张图片作为示例
#         #         break

# if __name__ == "__main__":
#     main()

    def save_replace_3rd_channel(self, path_loss, input_image, save_path, image_name):
        """Replace the third channel of input image with path loss and save as PNG"""
        # Create a copy of input_image to avoid modifying the original
        modified_image = input_image.copy()
        # Replace the third channel with path_loss
        modified_image[:,:,2] = path_loss
        save_file = os.path.join(save_path, f'{image_name}.png')
        Image.fromarray(modified_image.astype(np.uint8)).save(save_file)

    def visualize_combined_image2(self, image_path):
        """
        Visualize the saved image with path loss as the third channel
        
        Args:
            image_path: Path to the combined image
        """
        # Read the image
        combined_image = np.array(Image.open(image_path))
        
        # Separate channels
        reflectance = combined_image[:,:,0]    # 反射率
        transmittance = combined_image[:,:,1]   # 透射率
        path_loss = combined_image[:,:,2]       # 路径损耗 (now in the third channel)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # 1. Reflectance
        im0 = axes[0,0].imshow(reflectance, cmap='viridis')
        plt.colorbar(im0, ax=axes[0,0], label='Reflectance')
        axes[0,0].set_title('Channel 0: Reflectance')
        axes[0,0].axis('off')
        
        # 2. Transmittance
        im1 = axes[0,1].imshow(transmittance, cmap='viridis')
        plt.colorbar(im1, ax=axes[0,1], label='Transmittance')
        axes[0,1].set_title('Channel 1: Transmittance')
        axes[0,1].axis('off')
        
        # 3. Path Loss (now in third channel)
        im2 = axes[1,0].imshow(path_loss, cmap='jet', vmin=0, vmax=160)
        plt.colorbar(im2, ax=axes[1,0], label='Path Loss (dB)')
        axes[1,0].set_title('Channel 2: Path Loss')
        axes[1,0].axis('off')
        
        # 4. Keep the fourth subplot empty or use it for other visualization
        axes[1,1].axis('off')
        
        # Extract information from filename
        filename = os.path.basename(image_path)
        parts = filename.split('_')
        building_id = parts[0][1:]
        antenna_id = parts[1][3:]
        freq_id = parts[2][1:]
        sample_id = parts[3][1:].split('.')[0]
        
        plt.suptitle(
            f'Image Channels with Path Loss\n'
            f'Building {building_id}, Antenna {antenna_id}, '
            f'Freq {freq_id}, Sample {sample_id}',
            fontsize=14
        )
        
        plt.tight_layout()
        plt.show()

def main():
    from tqdm.auto import tqdm
    vis = False
    calculator = PathLossCalculator()
    
    input_dir = 'datasets/ICASSP2025_Dataset/Inputs/Task_3_ICASSP'
    save_dir = 'datasets/ICASSP2025_Dataset/Inputs/Task_3_ICASSP_path_loss_results_replaced'
    os.makedirs(save_dir, exist_ok=True)
    
    ii = 0
    for file_name in tqdm(os.listdir(input_dir)):
        ii += 1
        if file_name.endswith('.png'):
            parts = file_name.split('_')
            building_id = int(parts[0][1:])
            antenna_id = int(parts[1][3:])
            freq_id = int(parts[2][1:])
            sample_id = int(parts[3][1:].split('.')[0])
            
            path_loss, input_image = calculator.create_path_loss_map(
                building_id, antenna_id, freq_id, sample_id
            )
            
            if vis:
                # Visualize for checking
                calculator.visualize(path_loss, input_image, 
                                building_id, antenna_id, freq_id, sample_id)
                
            image_name = f'B{building_id}_Ant{antenna_id}_f{freq_id}_S{sample_id}'
            calculator.save_replace_3rd_channel(path_loss, input_image, save_dir, image_name)
            
            if vis:
                # Visualize saved image
                combined_path = os.path.join(save_dir, f'{image_name}.png')
                calculator.visualize_combined_image2(combined_path)
        
        # if vis and ii > 2:  # Only process first two images as example
        #     break

if __name__ == "__main__":
    main()
    
    
# find /home/work/test_data/HI-Diff-PL/antenna_vis/path_loss_results_replaced -name "*.png" -type f -printf '.' | wc -c