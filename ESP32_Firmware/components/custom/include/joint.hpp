#pragma once
#include <cstdint>
#include <cstring>  // for memcpy
#include <vector>

// ============ Constants ============ //
#define ARM_JOINT_NUM     5
#define NECK_JOINT_NUM    2
#define BOTTOM_JOINT_NUM  2
#define TOTAL_JOINT_NUM   (2 * ARM_JOINT_NUM + NECK_JOINT_NUM + BOTTOM_JOINT_NUM)

// ============ Basic Joint Struct ============ //
struct Joint {
    uint8_t id;            // Servo ID
    float expect_angle;    // Desired angle in degrees
    float current_angle;   // Measured/estimated angle
    float reset_angle;     // Reset/default angle
};

// ============ Group Structs ============ //
struct ArmGroup {
    Joint joints[ARM_JOINT_NUM];
};

struct NeckGroup {
    Joint joints[NECK_JOINT_NUM];
};

struct BottomGroup {
    Joint joints[BOTTOM_JOINT_NUM];
};

// ============ RobotJointState Class ============ //
class RobotJointState {
public:
    ArmGroup right_arm;
    ArmGroup left_arm;
    NeckGroup neck;
    BottomGroup bottom;

    RobotJointState() {
        init_default();
    }

    // Total number of joints (computed)
    inline uint8_t length() const {
        return TOTAL_JOINT_NUM;
    }

    // Set expect_angle by servo ID
    void setAngleById(uint8_t id, float angle_deg) {
        Joint* joint = findJointById(id);
        if (joint != nullptr) {
            joint->expect_angle = angle_deg;
        }
    }

    // Get all joints as a flat vector (useful for iteration)
    std::vector<Joint*> getAllJoints() {
        std::vector<Joint*> result;
        for (int i = 0; i < ARM_JOINT_NUM; ++i) {
            result.push_back(&left_arm.joints[i]);
            result.push_back(&right_arm.joints[i]);
        }
        for (int i = 0; i < NECK_JOINT_NUM; ++i)
            result.push_back(&neck.joints[i]);
        for (int i = 0; i < BOTTOM_JOINT_NUM; ++i)
            result.push_back(&bottom.joints[i]);
        return result;
    }

    // Reset all joints to their reset_angle
    void resetAllAngles() {
        for (Joint* joint : getAllJoints()) {
            joint->expect_angle = joint->reset_angle;
        }
    }

private:
    // Initialize default IDs and reset angles
    void init_default() {
        uint8_t id_counter = 0;
        for (int i = 0; i < ARM_JOINT_NUM; ++i) {
            left_arm.joints[i].id = id_counter++;
            right_arm.joints[i].id = id_counter++;
        }
        for (int i = 0; i < NECK_JOINT_NUM; ++i) {
            neck.joints[i].id = id_counter++;
        }
        for (int i = 0; i < BOTTOM_JOINT_NUM; ++i) {
            bottom.joints[i].id = id_counter++;
        }
    }

    // Helper: Find joint by ID
    Joint* findJointById(uint8_t id) {
        for (Joint* joint : getAllJoints()) {
            if (joint->id == id) return joint;
        }
        return nullptr;
    }
};
