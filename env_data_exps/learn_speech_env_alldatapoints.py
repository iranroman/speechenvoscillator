import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys 
import matplotlib.pyplot as plt
import os
import scipy.io
from scipy.signal import butter, lfilter, hilbert
from scipy.stats.stats import pearsonr

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

lr = 0.1

fs = 100
cutoff = 10
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# get name of all mat files in the directory
all_files = os.listdir('data')
all_env = [f for f in all_files if f[-1]=='t' and f[0] in ['1','2']]
all_env.sort()

# Load the mat files
data = [scipy.io.loadmat(''.join(['data/',f])) for f in all_env]

# preprocess the speech envelopes
all_envelopes = [np.squeeze(f['env_ds']) for f in data]
all_envelopes = [np.diff(butter_lowpass_filter(np.squeeze(np.array(env)),cutoff,fs), axis=0) for env in all_envelopes]
max_len = len(max(all_envelopes,key=len))+1
all_envelopes = np.array([np.pad(np.squeeze(env),(0,max_len-len(env))) for env in all_envelopes])
# create a mask to apply to the model's output
mask = np.ma.masked_where(all_envelopes[:,1:] != 0.0, all_envelopes[:,1:]).mask.astype(np.float32)
all_envelopes = (all_envelopes-np.expand_dims(np.mean(all_envelopes, axis=1),-1))/np.expand_dims(np.std(all_envelopes, axis=1),-1)/4

np.random.seed(0)
ntest = 20
ntrain = 180
rand_idx = np.random.choice(180,180,replace=False)

# input data dimensionality parameters
ndatapoints = ntrain
num_batches = 1
batch = 0
batch_size = int(ndatapoints/num_batches)


# declare the model's stimulus values
nchannels = 1 
stim_values = all_envelopes[rand_idx[ndatapoints*batch:ndatapoints*batch+ndatapoints],:,np.newaxis]
stim_values = tf.convert_to_tensor(stim_values, dtype=tf.float32) 
stim_values = tf.complex(stim_values, tf.zeros_like(stim_values))

# declare the model's target values
clean_values = np.cos(np.angle(hilbert(all_envelopes[:,1:],axis=1)))
clean_values = tf.convert_to_tensor(clean_values[rand_idx[ndatapoints*batch:ndatapoints*batch+ndatapoints]], dtype=tf.float32)

mask = mask[rand_idx[ndatapoints*batch:ndatapoints*batch+ndatapoints]]

##############################
# define the stimulus object #
##############################
class stimulus():

    def __init__(self, name = '',
                    values = tf.constant(0, dtype=tf.complex64, shape=(1,1,1)),
                    fs = tf.constant(1.0)):

        self.name = name
        self.values = values
        vshape = tf.shape(self.values)
        self.ndatapoints = vshape[0]
        self.nsamps = tf.cast(vshape[1],dtype=tf.float32)
        self.nchannels = vshape[2]
        self.fs = tf.constant(fs, dtype=tf.float32)
        self.dt = tf.constant(1.0/self.fs, dtype=tf.float32)
        self.dur = tf.constant(self.nsamps/self.fs, dtype=tf.float32)

################
s = stimulus(values = stim_values,
                fs = fs)



###############################
# layer of oscillators object #
###############################
class neurons():

    def __init__(self, name = '',
                    osctype = 'grfnn',
                    params = None,
                    freqs = None,
                    initconds = tf.constant(0, dtype=tf.complex64, shape=(256,))):

        self.name = name
        self.osctype = osctype
        self.params = params
        self.initconds = initconds
        self.params['freqs'] = freqs
        self.N = 1
        self.connections = []

# natural frequency of oscillation
flow = 4.5
fhigh = 4.5
Noctaves = np.log2(fhigh/flow)
Noscperoct = 12
N = int(Noscperoct*Noctaves+1)
#w0 = tf.Variable(2*np.pi*np.logspace(np.log10(flow), np.log10(fhigh),N), dtype=tf.float32)

# oscillator parameters and initial conditions
initconds = tf.constant(0.1+1j*0.0, dtype=tf.complex64, shape=(N,))
l_params_dict = {'alpha':tf.Variable(0.452, dtype=tf.float32, constraint=lambda z: tf.clip_by_value(z,0.0,np.inf)),
             'beta1':tf.Variable(-10.751, dtype=tf.float32, constraint=lambda z: tf.clip_by_value(z,-np.inf,0.0)), 
             'beta2':tf.constant(0.0, dtype=tf.float32),#constraint=lambda z: tf.clip_by_value(z,-np.inf,0.0)),
             'delta':tf.constant(0.0, dtype=tf.float32),
             'cz': tf.Variable(1.55, dtype=tf.float32),
             'cw': tf.Variable(1.768, dtype=tf.float32),
             'cr': tf.Variable(0.504, dtype=tf.float32, constraint=lambda z: tf.clip_by_value(z,0.0,np.inf)),
             'w0': tf.Variable(4.882*2*np.pi, dtype=tf.float32),
             'epsilon':tf.constant(1.0, dtype=tf.float32)}

################
layer1 = neurons(osctype = 'grfnn', 
                params = l_params_dict,
                freqs = [l_params_dict['w0']/(2*np.pi)], 
                initconds = initconds)


##################################
# Object and function to connect #
# stimulus and oscillator        #
##################################
class connection():

    def __init__(self, name = '',
                    source = None,
                    target = None,
                    matrixinit = None,
                    params = None):

        self.name = name
        self.source = source
        self.target = target
        self.params = params
        self.matrixinit = matrixinit
        self.params['freqss'] = tf.constant(0, dtype=tf.float32, shape=(self.source.nchannels,))
        self.params['freqst'] = self.target.params['freqs']
        self.params['typeint'] = None

def connect(connname = '', source = None, target = None, matrixinit = None, params = None):

    target.connections = [connection(name = connname,
                                source=source,
                                target=target,
                                matrixinit=matrixinit,
                                params=params)]

    return target

# connection parameters
l_conn_params_dict = {'weight':tf.constant([1.0]*N, dtype=tf.float32)}

########################################
layer1 = connect(source=s, target=layer1, 
                params=l_conn_params_dict,
                matrixinit=tf.constant(1.0+1j*0.0, 
                                        dtype=tf.complex64, shape=(nchannels,N)))


###########################
# ODE function definition #
###########################
def xdot_ydot(t, x_y, alpha, beta1, beta2, delta, cz, cw, cr, w0, epsilon, sources_state, dtype=tf.float32):

    # keep some parameters always positive
    omega = tf.constant(2*np.pi, dtype=dtype)

    x, y, freqs = tf.split(x_y, 3, axis=1)

    x2plusy2 = tf.add(tf.pow(x, 2),
                        tf.pow(y, 2))
    x2plusy2squared = tf.pow(x2plusy2, 2)
    HOT = tf.divide(
            tf.multiply(tf.multiply(epsilon,beta2),
                x2plusy2squared),
            tf.add(tf.constant(1.0, dtype=dtype),
                -tf.multiply(epsilon, x2plusy2)))

    xnew = tf.add_n([tf.multiply(alpha, x),
                        tf.multiply(omega, tf.multiply(-1.0, y)),
                        tf.multiply(tf.multiply(delta, tf.multiply(-1.0, y)), x2plusy2),
                        tf.multiply(beta1, tf.multiply(x, x2plusy2)),
                        tf.multiply(x, HOT)])

    ynew = tf.add_n([tf.multiply(alpha, y),
                        tf.multiply(omega, x),
                        tf.multiply(beta1, tf.multiply(y, x2plusy2)),
                        tf.multiply(tf.multiply(delta, x), x2plusy2),
                        tf.multiply(y, HOT)])

    # compute input
    sr, si = tf.split(sources_state[0], 2, axis=1)
    csr = tf.multiply(cz, sr)
    csi = tf.multiply(cz, si)
    csr = tf.multiply(csr, tf.add(tf.divide(1.0,tf.add(tf.pow(tf.add(1.0,-x),2),tf.pow(y,2))),
        -tf.divide(x,tf.add(tf.pow(tf.add(1.0,-x),2),tf.pow(y,2)))))
    csi = tf.multiply(csi, tf.multiply(-1.0, tf.divide(y,
            tf.add(tf.pow(tf.add(1.0,-x),2),tf.pow(y,2)))))

    xnew = tf.multiply(freqs, tf.add(xnew,csr))
    ynew = tf.multiply(freqs, tf.add(ynew,csi))
    xnew_ynew = tf.concat([xnew, ynew], axis=1)

    w = tf.multiply(freqs,2*np.pi)
    wnew = tf.add(
            -tf.divide(tf.multiply(cw,
                tf.multiply(tf.sin(tf.math.angle(tf.complex(x,y))),
            tf.multiply(sr, tf.add(tf.divide(1.0,tf.add(tf.pow(tf.add(1.0,-x),2),tf.pow(y,2))),
        -tf.divide(x,tf.add(tf.pow(tf.add(1.0,-x),2),tf.pow(y,2))))))), 
            tf.abs(tf.complex(x,y))),
            -tf.multiply(cr,tf.divide((w-w0),w0)))

    wnew = tf.multiply(freqs, wnew)
    dxdt_dydt = xnew_ynew
    dxdt_dydt_dwdt = tf.concat([dxdt_dydt, tf.divide(wnew,2*np.pi)], axis=1)

    return dxdt_dydt_dwdt

####################################
# Object to define the GrFNN model #
####################################
class Model():

    def __init__(self, name = '',
                    layers = None,
                    stim = None,
                    zfun = xdot_ydot):

        self.name = name
        self.layers = layers
        self.stim = stim
        self.zfun = zfun
        self.dt = self.stim.dt
        self.half_dt = self.dt/2
        self.nsamps = self.stim.nsamps
        self.dur = self.stim.dur
        self.time = tf.range(self.dur, delta=self.dt, dtype=tf.float32)

##########################
GrFNN = Model(layers=[layer1], stim=s)

def correlation(x, y, axis=1):    
    mx = tf.math.reduce_mean(x,axis=axis,keepdims=True)
    my = tf.math.reduce_mean(y,axis=axis,keepdims=True)
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym),axis=axis,keepdims=True) 
    r_den = tf.math.reduce_std(xm,axis=axis,keepdims=True) * tf.math.reduce_std(ym,axis=axis,keepdims=True)
    return 0.5 + (r_num / r_den)/2

def train_step(optim, target, mask, time, layers_state, layers_alpha, layers_beta1, layers_beta2, layers_delta, layers_cz, layers_cw, layers_cr, layers_w0, layers_epsilon, zfun, stim_values, dtype):
    with tf.GradientTape() as tape:
        mse = tf.losses.MeanSquaredError()
        # keep some parameters always positive
        layers_states = Runge_Kutta_4(time, layers_state, layers_alpha, layers_beta1, layers_beta2, layers_delta, layers_cz, layers_cw, layers_cr, layers_w0, layers_epsilon, zfun, stim_values, dtype)

        l_output_r, l_output_i, freqs = tf.split(layers_states[0],3,axis=2) 
        l_output_r = tf.transpose(l_output_r,(1,2,0))
        l_output_i = tf.transpose(l_output_i,(1,2,0))
        l_z = tf.complex(l_output_r,l_output_i)
        l_z = tf.cos(tf.math.angle(l_z))
        freqs = tf.transpose(freqs,(1,2,0))
        l_z = tf.squeeze(l_z,axis=1)
        cleaned = tf.multiply(l_z,mask)
        #curr_loss = mse(target, cleaned) +
        curr_loss = tf.reduce_mean(-tf.math.log(correlation(target, cleaned)))#+ tf.reduce_mean(-tf.math.log(correlation(tf.experimental.numpy.diff(target),tf.experimental.numpy.diff(cleaned))))
    tf.print('==========================')
    tf.print('    Loss: ', curr_loss)
    tf.print('==========================')
    var_list = {
        'alpha ':layers_alpha, 
        'beta1 ': layers_beta1, 
        #'beta2 ': layers_beta2, 
        #'delta ': layers_delta, 
        'cz ': layers_cz, 
        'cw ': layers_cw, 
        'cr ': layers_cr, 
        'w0 ': layers_w0
    }
    grads = tape.gradient(curr_loss, list(var_list.values()))
    optim.apply_gradients(zip(grads, list(var_list.values())))
    return layers_states, tf.squeeze(l_output_r,axis=1), freqs, curr_loss, var_list


@tf.function()
def train_GrFNN(optim, target, mask, time, layers_state, layers_alpha, layers_beta1, layers_beta2, layers_delta, layers_cz, layers_cw, layers_cr, layers_w0, layers_epsilon, zfun, stim_values, dtype):
    layers_states = train_step(optim, target, mask, time, layers_state, layers_alpha, layers_beta1, layers_beta2, layers_delta, layers_cz, layers_cw, layers_cr, layers_w0, layers_epsilon, zfun, stim_values, dtype)
    return layers_states

##########
# solver #
##########
def complex2concat(x, axis):
    return tf.concat([tf.math.real(x),tf.math.imag(x)],axis=axis)
def Runge_Kutta_4(time, layers_state, layers_alpha, layers_beta1,
                layers_beta2, layers_delta, layers_cz, layers_cw, layers_cr, layers_w0, layers_epsilon, zfun, stim_values, dtype=tf.float16):

    def scan_fn(layers_state, time_dts_stim):

        def get_next_k(time_val, layers_state):

            layers_k = [zfun(time_val, layer_state, layers_alpha, layers_beta1, layers_beta2, layers_delta, layers_cz, layers_cw, layers_cr, layers_w0, layers_epsilon, layers_state, dtype)
                for layer_state in layers_state[1:]]

            return layers_k

        def update_states(time_scaling, layers_k0, layers_k, new_stim):

            layers_state = [tf.add(layer_k0, tf.scalar_mul(time_scaling, layer_k))
                        for (layer_k0, layer_k) in zip(layers_k0, layers_k)]
            layers_state.insert(0, new_stim)

            return layers_state

        t, dt, stim, stim_shift = time_dts_stim

        t_plus_half_dt = tf.add(t, dt/2)
        t_plus_dt = tf.add(t, dt)

        layers_k0 = layers_state.copy()
        layers_state.insert(0, stim)

        layers_k1 = get_next_k(t, layers_state)
        layers_state = update_states(dt/2, layers_k0, layers_k1,
                                tf.divide(tf.add(stim, stim_shift),2))
        layers_k2 = get_next_k(t_plus_half_dt, layers_state)
        layers_state = update_states(dt/2, layers_k0, layers_k2,
                                tf.divide(tf.add(stim, stim_shift),2))
        layers_k3 = get_next_k(t_plus_half_dt, layers_state)
        layers_state = update_states(dt, layers_k0, layers_k3,
                                stim_shift)
        layers_k4 = get_next_k(t_plus_dt, layers_state)

        layers_state = [tf.add(layer_k0,
                    tf.multiply(dt/6,  tf.add_n([layer_k1,
                                                tf.scalar_mul(2, layer_k2),
                                                tf.scalar_mul(2, layer_k3),
                                                layer_k4])))
                        for (layer_k0, layer_k1, layer_k2, layer_k3, layer_k4)
                        in zip(layers_k0, layers_k1, layers_k2, layers_k3, layers_k4)]
        #tf.print('=============')
        #tf.print(layers_state)

        return layers_state

    dts = time[1:] - time[:-1]
    layers_states = tf.scan(scan_fn,
                        [time[:-1],
                            dts,
                            tf.transpose(stim_values[:,:-1,:],(1,0,2)),
                            tf.transpose(stim_values[:,1:,:],(1,0,2))],
                        layers_state)

    return layers_states


def get_model_variables_for_integration(Model, dtype=tf.float16):

    time = tf.cast(Model.time,dtype)
    stim_values = tf.cast(complex2concat(Model.stim.values,2),dtype)
    layer_state = [tf.tile(tf.expand_dims(tf.cast(tf.concat([complex2concat(layer.initconds,0), [layer.params["w0"]/(2*np.pi)]],axis=0),dtype),axis=0),
                        tf.constant([Model.stim.ndatapoints.numpy(),1]))
                    for layer in Model.layers]
    layer_alpha = [tf.cast(layer.params['alpha'],dtype) for layer in Model.layers][0]
    layer_beta1 = [tf.cast(layer.params['beta1'],dtype) for layer in Model.layers][0]
    layer_beta2 = [tf.cast(layer.params['beta2'],dtype) for layer in Model.layers][0]
    layer_delta = [tf.cast(layer.params['delta'],dtype) for layer in Model.layers][0]
    layer_cz = [tf.cast(layer.params['cz'],dtype) for layer in Model.layers][0]
    layer_cw = [tf.cast(layer.params['cw'],dtype) for layer in Model.layers][0]
    layer_cr = [tf.cast(layer.params['cr'],dtype) for layer in Model.layers][0]
    layer_w0 = [tf.cast(layer.params['w0'],dtype) for layer in Model.layers][0]
    layer_epsilon = [tf.cast(layer.params['epsilon'],dtype) for layer in Model.layers][0]
    zfun = Model.zfun

    return layer_state, layer_alpha, layer_beta1, layer_beta2, layer_delta, layer_cz, layer_cw, layer_cr, layer_w0, layer_epsilon, zfun, stim_values, time

def correlation_numpy(x, y, axis=1):    
    return [pearsonr(i,j)[0] for i,j in zip(x,y)]
#    mx = np.mean(x,axis=axis,keepdims=True)
#    my = np.mean(y,axis=axis,keepdims=True)
#    xm, ym = x-mx, y-my
#    r_num = np.mean(tf.multiply(xm,ym),axis=axis,keepdims=True) 
#    r_den = np.std(xm,axis=axis,keepdims=True) * np.std(ym,axis=axis,keepdims=True)
#    return r_num / r_den

# let's integrate and train
num_epochs = 1
var_list_old = {}
optim = tf.optimizers.Adam(lr)
for e in range(num_epochs):

    print('==========================')
    print('==========================')
    print("Epoch: ", e+1)
    print('--------------------------')
    # get variables for integration
    layers_state, layers_alpha, layers_beta1, layers_beta2, layers_delta, layers_cz, layers_cw, layers_cr, layers_w0, layers_epsilon, zfun, input_values, time = get_model_variables_for_integration(GrFNN, tf.float32)

    #rand_idx = np.random.choice(ndatapoints,ndatapoints,replace=False)
    clean_batches = clean_values
    stim_batches = input_values
    print('alpha:    ', layers_alpha.numpy())
    print('beta1:    ', layers_beta1.numpy())
    print('beta2:    ', layers_beta2.numpy())
    print('delta:    ', layers_delta.numpy())
    print('cz:       ', layers_cz.numpy())
    print('cw:       ', layers_cw.numpy())
    print('cr:       ', layers_cr.numpy())
    print('w0:       ', layers_w0.numpy())
    print('w0/(2*pi):', layers_w0.numpy()/(2*np.pi))
    for ibatch in range(num_batches):
        #tf.print('-- batch: ', ibatch+1, ' (', num_batches, ')')
        batch_clean = clean_batches[int(ibatch*batch_size):int(batch_size+ibatch*batch_size)]
        batch_stim = stim_batches[int(ibatch*batch_size):int(batch_size+ibatch*batch_size)]
        layers_states, cleaned, frq, loss, var_list = train_GrFNN(optim, batch_clean, mask, time, layers_state, layers_alpha, layers_beta1, layers_beta2, layers_delta, layers_cz, layers_cw, layers_cr, layers_w0, layers_epsilon, zfun, batch_stim, tf.float32)
    plt.rcParams["figure.figsize"] = (20,10)
    plt.grid()
    plt.plot(GrFNN.time[:-1],np.squeeze(all_envelopes[rand_idx[0],1:]))
    plt.plot(GrFNN.time[:-1],np.squeeze(cleaned[0]))
    plt.plot(GrFNN.time[:-1],np.squeeze(frq[0]), '--')
    plt.title('corr: '+"{:.4f}".format(np.mean(correlation_numpy(all_envelopes[rand_idx[ndatapoints*batch:ndatapoints*batch+ndatapoints],1:],cleaned.numpy())))+', loss: '+"{:.4f}".format(loss.numpy())+'   '+', '.join([k+"{:.3f}".format(v.numpy()) for k, v in var_list_old.items()]))
    plt.savefig("epoch"+str(e+1)+".png")
    plt.close()
    var_list_old = var_list
    np.save('envelopes.npy',all_envelopes)
    np.save('model_output.npy',cleaned)
    print(cleaned.shape)
    print(all_envelopes.shape)
