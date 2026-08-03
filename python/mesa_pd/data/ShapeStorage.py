# -*- coding: utf-8 -*-

from ..utility import generate_file


class ShapeStorage:
    def __init__(self, particle_storage):
        particle_storage.add_property("shapeID", "size_t", defValue="", syncMode="ON_GHOST_CREATION")
        particle_storage.add_property("rotation", "walberla::mesa_pd::Rot3", defValue="", syncMode="ALWAYS")
        particle_storage.add_property("angularVelocity", "walberla::mesa_pd::Vec3", defValue="0_r",
                                      syncMode="ALWAYS")
        particle_storage.add_property("torque", "walberla::mesa_pd::Vec3", defValue="0_r", syncMode="NEVER")

        self.ps = particle_storage

    def generate(self, module):
        ctx = {'module': module, 'particle': self.ps.get_context()}

        generate_file(module['module_path'], 'data/ShapeStorage.templ.h', ctx)
