from numba.np.ufunc import parallel
import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
# from numba import njit, prange, set_num_threads

EVENT_TYPE = np.dtype(
    [("t", "f8"), ("x", "u2"), ("y", "u2"), ("p", "b")], align=True
)

TOL = .08

CONFIG = SimpleNamespace(
    **{
        "contrast_thresholds": (0.01, 0.01),
        "sigma_contrast_thresholds": (0.0, 0.0),
        "refractory_period_ns": 1000,
        "max_events_per_frame": 600000,
    }
)



def generate_image_for_sim(blinks_distance = 3.5, N_ph = 100, sigma = 4.8, add_noise=True, N_ph2=None, noise_level=None):

    def gaussian(x, var): 
        return np.exp(- x**2 / (2*var))
    def gaussian_shift(x, center, var): 
        return np.exp(- (x - center)**2 / (2*var))

    vec = np.arange(-10, 11)
    X, Y = np.meshgrid(vec,vec)
    g = gaussian(vec, sigma)
    g_shift = gaussian_shift(vec, blinks_distance, sigma)

    input_img_1 = N_ph * g.reshape(-1,1) * g.reshape(1,-1) + 0.001
    if N_ph2 is None:
        input_img_2 = N_ph * g_shift.reshape(-1,1) * g_shift.reshape(1,-1) + 0.001
    else:
        input_img_2 = N_ph2 * g_shift.reshape(-1,1) * g_shift.reshape(1,-1) + 0.001

    input_img_1[input_img_1<0.1] = 0.001
    input_img_2[input_img_2<0.1] = 0.001

    if add_noise:
        if noise_level: noise_coeff = noise_level
        else: noise_coeff = 0.03
        n1 = input_img_1>0.001
        noise = n1 * np.random.poisson(8, size=input_img_1.shape)
        input_img_1 += noise + np.random.poisson(noise_coeff, size=input_img_1.shape)
        n2 = input_img_2>0.001
        noise = n2 * np.random.poisson(8, size=input_img_1.shape)
        input_img_2 += noise + np.random.poisson(noise_coeff, size=input_img_1.shape)
        
    input_img = input_img_1 + input_img_2 - 0.001
    image_sequence = [input_img_1, input_img, input_img_2, np.zeros_like(input_img)+0.001]
    time_sequence = [1e-3, 2e-3, 3e-3, 4e-3]

    return image_sequence, time_sequence

def plot_bar3d_from_2d_image(intensities):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    
    fig.set_facecolor('none')
    ax = fig.add_subplot(projection='3d')

    # set x, y, and z coordinates for the bars
    xpos, ypos = np.meshgrid(range(intensities.shape[0]), range(intensities.shape[1]))
    xpos = xpos.flatten()   # make 1D
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    # set dimensions of bars
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = intensities.flatten()
    colors = plt.cm.viridis(dz/dz.max()) 
    img = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, alpha=1, color=colors, edgecolor='gray')

    # set axis labels and title
    ax.set_xlabel('x position, (a.u.)')
    ax.set_ylabel('y position, (a.u.)')
    ax.set_zlabel('Intensity, (a.u.)')

def generate_single_frame(events, input_image):
    frame = np.zeros(input_image.shape)
    for id in np.arange(len(events)):
        if events[id]['p'] == 1:
            frame[events[id]['y'],events[id]['x']]+=1.
            # frame[events[id]['y'],events[id]['x']]*=1.02
        else: 
            frame[events[id]['y'],events[id]['x']]-=1.
            # frame[events[id]['y'],events[id]['x']]*=1.02
    return frame

# @njit(parallel=True)
def esim(
    x_end,
    current_image,
    previous_image,
    delta_time,
    crossings,
    last_time,
    output_events,
    spikes,
    refractory_period_ns,
    max_events_per_frame,
    n_pix_row,
):
    count = 0
    # max_spikes = int(delta_time / (refractory_period_ns * 1e-3))
    max_spikes = 10e6
    for x in np.arange(x_end):
        itdt = np.log(current_image[x])
        it = np.log(previous_image[x])
        deltaL = itdt - it

        if np.abs(deltaL) < TOL:
            continue

        polarity = np.sign(deltaL)
        # num_events = int(deltaL / THRSHLD)
        cross_update = polarity * TOL
        crossings[x] = np.log(crossings[x]) + cross_update

        lb = crossings[x] - it
        ub = crossings[x] - itdt

        pos_check = lb > 0 and (polarity == 1) and ub < 0
        neg_check = lb < 0 and (polarity == -1) and ub > 0

        spike_nums = (itdt - crossings[x]) / TOL
        cross_check = pos_check + neg_check
        spike_nums = np.abs(int(spike_nums * cross_check))

        crossings[x] = itdt - cross_update
        if spike_nums > 0:
            spikes[x] = polarity

        spike_nums = max_spikes if spike_nums > max_spikes else spike_nums

        current_time = last_time
        for i in range(spike_nums):
            output_events[count]['x'] = x % n_pix_row
            output_events[count]['y'] = x // n_pix_row
            output_events[count]['t'] = np.round(current_time * 1e-6, 6)
            output_events[count]['p'] = 1 if polarity > 0 else -1

            count += 1
            current_time += (delta_time) / spike_nums

            if count == max_events_per_frame:
                return count

    return count

class EventSimulator:
    def __init__(self, W, H, first_image=None, first_time=None, config=CONFIG):
        self.H = H
        self.W = W
        self.config = config
        self.last_image = None
        if first_image is not None:
            assert first_time is not None
            self.init(first_image, first_time)

        self.npix = H * W

    def init(self, first_image, first_time):
        # print("Initialized event camera simulator with sensor size:", first_image.shape)

        self.resolution = first_image.shape  # The resolution of the image

        # We ignore the 2D nature of the problem as it is not relevant here
        # It makes multi-core processing more straightforward
        first_image = first_image.reshape(-1)

        # Allocations
        self.last_image = first_image.copy()
        self.current_image = first_image.copy()

        self.last_time = first_time

        self.output_events = np.zeros(
            (self.config.max_events_per_frame), dtype=EVENT_TYPE
        )
        self.event_count = 0
        self.spikes = np.zeros((self.npix))

    def convert_event_img_rgb(self, image):
        image = image.reshape(self.H, self.W)
        out = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        out[:, :, 0] = np.clip(image, 0, 1) * 255
        out[:, :, 2] = np.clip(image, -1, 0) * -255
        return out

    def convert_event_img_gray(self, image):
        image = image.reshape(self.H, self.W)
        out = np.zeros((self.H, self.W), dtype=np.uint8) + 127
        # out[:, :] = np.clip(image, 0, 1) * 255
        # out[:, :] = np.clip(image, -1, 0) * 0
        out[image > 0] = 255
        out[np.where(image < 0)] = 0
        return out

    def image_callback(self, new_image, new_time):
        if self.last_image is None:
            self.init(new_image, new_time)
            return None, None

        assert new_time > 0
        assert new_image.shape == self.resolution
        new_image = new_image.reshape(-1)  # Free operation

        np.copyto(self.current_image, new_image)

        delta_time = new_time - self.last_time

        config = self.config
        self.output_events = np.zeros(
            (self.config.max_events_per_frame), dtype=EVENT_TYPE
        )
        self.spikes = np.zeros((self.npix))

        self.crossings = self.last_image.copy()
        self.event_count = esim(
            self.current_image.size,
            self.current_image,
            self.last_image,
            delta_time,
            self.crossings,
            self.last_time,
            self.output_events,
            self.spikes,
            config.refractory_period_ns,
            config.max_events_per_frame,
            self.W,
        )

        np.copyto(self.last_image, self.current_image)
        self.last_time = new_time

        result = self.output_events[: self.event_count]
        result.sort(order=["t"], axis=0)

        return self.spikes, result
    


def plot_event_sim_to_2gaussian(image_sequence, time_sequence, EventSimulator):
    
    fig, _ = plt.subplots(2, 4, figsize=(10, 5))
    fig.tight_layout()
    plt.rcParams['axes.grid'] = False
    EventSimulator.image_callback(np.zeros_like(image_sequence[0])+0.001, 0)
    
    l=0
    for id, i in enumerate(image_sequence):
        plt.subplot(2,4,l+1)
        maxval = np.amax(i)
        if maxval < 1: maxval = 100
        # if id==2: imm = plt.imshow(i, cmap='gray', vmax=60, vmin=0)
        imm = plt.imshow(i, cmap='gray', vmax=maxval, vmin=0)
        plt.xticks(np.arange(0, 21, 5))
        plt.yticks(np.arange(0, 21, 5))

        if id==3:
            cbar = fig.colorbar(imm, fraction=0.029, pad=-0.0001, aspect=33.5)
            cbar.ax.get_yaxis().labelpad = 11
            cbar.ax.set_ylabel('Intensity, (a.u.)', rotation=270)
            cbar.outline.set_visible(False)

        event_img, events = EventSimulator.image_callback(i, time_sequence[id])
        plt.subplot(2,4,l+1+4)
        if events is not None:
            im_ev2 = generate_single_frame(events, i)
            mask = (im_ev2 < 0)
            im_ev2[mask] -= 40

            mask2 = (im_ev2 > 0)
            im_ev2[mask2] += 40

            mask = (im_ev2 < -140)
            im_ev2[mask] += 60
            mask2 = (im_ev2 > 140)
            im_ev2[mask2] -= 60

            imm2 = plt.imshow(im_ev2, cmap='gray', vmax=150, vmin=-150)

            plt.xticks(np.arange(0, 21, 5))
            plt.yticks(np.arange(0, 21, 5))

            if id==3:
                cbar = fig.colorbar(imm2, fraction=0.029, pad=-0.0001, aspect=33.5)
                cbar.ax.get_yaxis().labelpad = 11
                cbar.ax.set_ylabel('# of events, (a.u.)', rotation=270)
                cbar.outline.set_visible(False)
        l+=1
    
    return fig