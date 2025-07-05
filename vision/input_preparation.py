import tensorflow as tf

from prep import get_all_image_samples
from prep import images_properties
from prep import filtering_images
from prep import images_relationships


def shp(f, shape):
    f.set_shape(shape)
    return f

def z_score_norm(x):
    u = tf.reduce_mean(x, axis=0)

    s = tf.math.reduce_std(x, axis=0)

    imgs_histograms_moments = (x - u) / s

    return imgs_histograms_moments, u, s

def z_score_norm_trans(x, y):
    u, s = y
    return (x - u) / s

def inp_prep_f(zipped,resh,resw,sdaq_df,locs_df,batch_sz):


    # for i in zipped.take(1):
    #     print(i[1]['loc'].numpy().decode())

    locs_ds = tf.data.Dataset.from_tensor_slices(locs_df)

    gaiso = get_all_image_samples.get_all_img_samples_c()

    all_imgs = gaiso.read_and_get_all(resh, resw, sdaq_df, locs_ds)  # all_metad_df['loc'])

    print(all_imgs.shape)

    num_bins = 256
    channelq = all_imgs.shape[-1]
    ipo = images_properties.images_properties_c(channelq, num_bins)

    all_imgs = tf.data.Dataset.from_tensor_slices(all_imgs)

    zipped = tf.data.Dataset.zip((all_imgs, zipped)).map(lambda a, b: (a,
                                                                       b[0],
                                                                       b[1]))

    zipped = zipped.map(lambda a, b, c: (shp(a, a.shape), shp(b, b.shape), {k: shp(v, v.shape) for k, v in c.items()})
                        )

    zipped = zipped.shuffle(zipped.cardinality(), seed=42)


    zipped = zipped.map(lambda a, b, c: (a,
                                         b,
                                         c,
                                         tf.numpy_function(ipo.compute_histogram, [a], Tout=tf.float32)),
                        num_parallel_calls=tf.data.AUTOTUNE)

    q = zipped.cardinality()

    zipped = zipped.map(lambda a, b, c, d: (a,
                                            b,
                                            c,
                                            ipo.compute_histograms_moments(d)),
                        # tf.numpy_function(ipo.compute_histograms_moments,[d],Tout=tf.float32)),
                        num_parallel_calls=tf.data.AUTOTUNE).batch(batch_sz)  # all_metad_df.shape[0]#q

    # def z_score_norm(x):
    #     u = tf.reduce_mean(x, axis=0)
    #
    #     s = tf.math.reduce_std(x, axis=0)
    #
    #     imgs_histograms_moments = (x - u) / s
    #
    #     return imgs_histograms_moments, u, s

    zipped = zipped.map(lambda a, b, c, d: (a,
                                            b,
                                            c,
                                            z_score_norm(d)
                                            )
                        )

    stat2_m = zipped.map(lambda a, b, c, d: (d[1], d[2]))

    zipped = zipped.map(lambda a, b, c, d: (a,
                                            b,
                                            c,
                                            d[0]
                                            )
                        ).unbatch()

    zipped = zipped.apply(tf.data.experimental.assert_cardinality(q))

    fio = filtering_images.filtering_images_c()

    zipped = zipped.map(lambda a, b, c, d: (a,
                                            b,
                                            c,
                                            d,
                                            shp(tf.numpy_function(fio.ImagePreprocessor, [a, d, batch_sz],
                                                                  # zipped.cardinality()
                                                                  Tout=tf.uint8), a.shape)  # ,[tf.uint8,tf.uint8]
                                            )
                        )  # ,num_parallel_calls=tf.data.AUTOTUNE)

    import matplotlib.pyplot as plt

    for org, _, _, _, image in zipped.skip(25).take(5):
        plt.subplot(2, 1, 1)
        plt.imshow(image)
        d_input_shape = image.shape
        plt.subplot(2, 1, 2)
        plt.imshow(org)
        plt.show()

    zipped = zipped.map(lambda a, b, c, d, e: (e,
                                               b,
                                               c,
                                               d
                                               )
                        )

    zipped = zipped.map(lambda a, b, c, d: (a,
                                            b,
                                            c,
                                            tf.numpy_function(ipo.compute_histogram, [a], Tout=tf.float32)
                                            )
                        , num_parallel_calls=tf.data.AUTOTUNE)

    zipped = zipped.map(lambda a, b, c, d: (a,
                                            b,
                                            c,
                                            d,
                                            ipo.compute_histograms_moments(d)
                                            )
                        , num_parallel_calls=tf.data.AUTOTUNE).batch(batch_sz)  # all_metad_df.shape[0]

    # def z_score_norm_trans(x, y):
    #     u, s = y
    #
    #     return (x - u) / s

    zipped = tf.data.Dataset.zip((zipped, stat2_m)).map(lambda a, b: (a[0],
                                                                      a[1],
                                                                      a[2],
                                                                      a[3],
                                                                      z_score_norm_trans(a[4], b)
                                                                      )
                                                        )  # .unbatch()

    iro = images_relationships.images_relationships_c()

    zipped = zipped.map(lambda a, b, c, d, e: (a,
                                               b,
                                               c,
                                               tf.reshape(d, [tf.shape(d)[0], -1]),
                                               )
                        )

    zipped = zipped.map(lambda a, b, c, d: (a,
                                            b,
                                            c,
                                            tf.cast(
                                                tf.numpy_function(iro.compute_histogram_correlation, [d],
                                                                  Tout=tf.float64),
                                                tf.float32)
                                            )
                        )

    # zipped = zipped.map(lambda a, b, c, d: (a,
    #                                         b,
    #                                         c,
    #                                         d,
    #                                         tf.numpy_function(iro.anomaly_detection, [d], Tout=tf.bool)
    #                                         )
    #                     )
    #
    # zipped = zipped.map(lambda a, b, c, d, e: (a,
    #                                            b,
    #                                            c,
    #                                            tf.boolean_mask(d, mask=tf.reshape(e, [-1]), axis=1),
    #                                            e
    #                                            )
    #                     )

    # for i in zipped:
    #     sumy=tf.math.reduce_sum(tf.cast(i[7],tf.int32))

    # sumy=zipped.map(lambda a,b,c,d,e,f,g,h:tf.math.reduce_sum(tf.cast(h,tf.int32)))

    # for i in sumy:
    #     print(i)
    # for i in zipped:
    #     print(i)

    # zipped = zipped.unbatch()#o

    # zipped=zipped.apply(tf.data.experimental.assert_cardinality(q))

    #
    # zipped = zipped.filter(lambda a, b, c, d, e: e)
    #
    # print('counting..')
    #
    # q_rs = zipped.reduce(np.int32(0), lambda d, _: d + 1).numpy()

    # o
    # zipped = zipped.apply(tf.data.experimental.assert_cardinality(q))#q_rs
    #
    #
    # zipped = zipped.map(lambda a, b, c, d: (tf.cast(a, tf.float32),
    #                                            b,
    #                                            c,
    #                                            tf.cast(d, tf.float32)
    #                                            )
    #                     )#e
    # o

    # o.batch(batch_sz)
    zipped = zipped.map(lambda a, b, c, d: (a,
                                            b,
                                            c,
                                            tf.cast(z_score_norm(d)[0], tf.float32)
                                            )
                        )  # zipped.cardinality()

    # iro.cluster_quantity_test(correlation_matrix)

    num_clusters = tf.constant(1, dtype=tf.int32)

    zipped = zipped.map(lambda a, b, c, d: (a, b, c, d,
                                            tuple(
                                                tf.numpy_function(iro.cluster_images_kmeans,
                                                                  [d, num_clusters],
                                                                  Tout=[tf.int32, tf.float32, tf.float64])
                                            )
                                            )
                        )

    zipped_pca = zipped.map(lambda a, b, c, d, e:
                            tf.numpy_function(iro.pca, [d, e[1], e[2]], Tout=[tf.float32,tf.float32,tf.float64])
                            )



    print('pca')
    t = ['Image Samples Correlations', f'Its {num_clusters} Clusters Centroids']

    import pandas as pd

    for datas,centroid,wcss in zipped_pca:  # .map(lambda a,b,c,d,e,f,g,h,i,j:a) :

        # pc_df1 = pd.DataFrame(data=datas, columns=['PC1', 'PC2'])
        #
        # pc_df2 = pd.DataFrame(data=centroid, columns=['PC1', 'PC2'])

        fig, ax = plt.subplots(figsize=(10, 8))  # plt.figure(figsize=(10, 8))

        ax.scatter(datas[:,0], datas[:,1], c='blue', label=f'{t[0]}')  # s

        ax.scatter(centroid[:,0], centroid[:,1], c='red', label=f'{t[1]}')  # s

        ax.set_title(f'2D PCA of Images Correlations also Wcss is {wcss}')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

        print('2D pca has created')  # samq=i[0].shape[0]

    return zipped,q,d_input_shape