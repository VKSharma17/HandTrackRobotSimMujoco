import xml.etree.ElementTree as ET
import os

def test():
    base_dir = "franka_emika_panda"
    xml_path = os.path.join(base_dir, "panda.xml")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    default_el = root.find("default")
    finger_joint = default_el.find(".//default[@class='finger']/joint")
    if finger_joint is not None:
        print("Found finger joint default!")
        print("Original attribs:", finger_joint.attrib)
        
        finger_joint.attrib["damping"] = "0.05"
        finger_joint.attrib["armature"] = "0.001"
        finger_joint.attrib["solimplimit"] = "0.99 0.999 0.001"
        finger_joint.attrib["solreflimit"] = "0.001 1"
        
        print("Modified attribs:", finger_joint.attrib)
    else:
        print("Finger joint default NOT found!")

if __name__ == "__main__":
    test()
