// Reproduce the parentless-keyframe graph seen in the real GDB backtrace.
#include <KeyFrame.h>
#include <Map.h>
#include <iostream>

int main() {
    ORB_SLAM3::Map map(0);
    ORB_SLAM3::KeyFrame orphan, child;
    orphan.UpdateMap(&map);
    child.UpdateMap(&map);
    orphan.mnId = 42;
    child.mnId = 43;
    orphan.AddChild(&child);
    // Upstream tries child.ChangeParent(nullptr) here, then dereferences null.
    orphan.SetBadFlag();
    if (orphan.isBad() || !orphan.hasChild(&child) || child.GetParent()) return 1;
    std::cout << "PASS: parentless keyframe retained without corrupting children\n";
    return 0;
}
