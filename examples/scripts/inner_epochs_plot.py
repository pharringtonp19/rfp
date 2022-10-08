# fig, ax = plt.subplots(dpi=300, tight_layout=True)
#     ax = plt.figure().gca()
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     for v, l in enumerate([0., 0.9, 1.]):
#         r[v] = []
#         for i in range(20):
#             inner_yuri = Trainer(loss_fn = supervised_loss,
#                         opt = optax.sgd(learning_rate=FLAGS.learning_rate),
#                         epochs = i)

#             cluster_loss = Cluster_Loss(inner_yuri=inner_yuri,
#                                         reg_value=v,
#                                         aux_status=False)
#             loss = cluster_loss(params, data)
#             r[v].append(loss)
#             print(cluster_loss(params, data))


#         ax.plot(jnp.arange(20), results, label=l)
#     plt.title('Loss', size=14, loc='left')
#     plt.xlabel('Inner Epochs', size=14)
#     plt.legend(frameon=False)
#     plt.show()
