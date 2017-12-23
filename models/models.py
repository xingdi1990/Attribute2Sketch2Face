def create_model(opt):
    model = None
    print(opt.model_name)
    if opt.model_name == 'attr_sketch_face':
        from .attr_sketch_face_model import attrsketchfaceModel
        model = attrsketchfaceModel()

    else:
        raise ValueError("Model [%s] not recognized." % opt.model_name)

    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
