<?php
include 'connect.php';

$libPathEncrypt = 'D:/Pribadi/VsCode/gatau apaya/xampp/htdocs/hackathon/encrypt.dll';
$libPathDecrypt = 'D:/Pribadi/VsCode/gatau apaya/xampp/htdocs/hackathon/decrypt.dll';

// Memuat shared library menggunakan FFI
$encrypt = \FFI::cdef("void encryptAES(const unsigned char *plainText, const unsigned char *key, unsigned char *cipherText);", $libPathEncrypt);
$dataLoginUsername = $_POST['username'];
$dataLoginPassword = $_POST['password'];
// Inisialisasi array untuk menyimpan data
if($dataLoginUsername == "admin" && $dataLoginPassword == "admin"){
    $result = mysqli_query($conn, "SELECT name_admin, username, password FROM admin");
    $usernames = array();
    $passwords = array();
    $keys = array();

    // Menyimpan data dari hasil query dalam masing-masing array
    while ($row = mysqli_fetch_assoc($result)) {
        $keys[] = $row['name_admin'];
        $usernames[] = $row['username'];
        $passwords[] = $row['password'];
    }
}
else{
    $result = mysqli_query($conn, "SELECT name_user, username, password, role_pekerjaan FROM user");
    $usernames = array();
    $passwords = array();
    $roles = array();
    $keys = array();

    // Menyimpan data dari hasil query dalam masing-masing array
    while ($row = mysqli_fetch_assoc($result)) {
        $keys[] = $row['name_user'];
        $usernames[] = $row['username'];
        $passwords[] = $row['password'];
        $roles[] = $row['role_pekerjaan'];
    }
}

// Membebaskan sumber daya hasil query
mysqli_free_result($result);

foreach ($usernames as $index => $username) {
    if($username == $dataLoginUsername && $passwords[$index] == $dataLoginPassword){
        $key = $keys[$index];

        // Enkripsi nickname (username) menggunakan fungsi encryptAES
        $keyData = FFI::new("unsigned char[" . strlen($key) . "]");
        FFI::memcpy($keyData, $key, strlen($key));
    
        $plainTextCData = FFI::new("unsigned char[" . strlen($username) . "]");
        FFI::memcpy($plainTextCData, $username, strlen($username));
    
        $cipherText = FFI::new("unsigned char[16]");
    
        $encrypt->encryptAES($plainTextCData, $keyData, $cipherText);
    
        $usernameEncryptResult = NULL;
        echo "Username: " . $username . "<br>";
        echo "Hasil Enkripsi Username: ";
        for ($i = 0; $i < 16; ++$i) {
            $usernameEncryptResult .= sprintf('%02x', $cipherText[$i]);
        }
        echo $usernameEncryptResult;
        echo "<br>";
    
        // Enkripsi password menggunakan fungsi encryptAES dengan key yang sama
        $keyDataPassword = FFI::new("unsigned char[" . strlen($key) . "]");
        FFI::memcpy($keyDataPassword, $key, strlen($key));
    
        $plainTextCDataPassword = FFI::new("unsigned char[" . strlen($passwords[$index]) . "]");
        FFI::memcpy($plainTextCDataPassword, $passwords[$index], strlen($passwords[$index]));
    
        $cipherTextPassword = FFI::new("unsigned char[16]");
    
        $encrypt->encryptAES($plainTextCDataPassword, $keyDataPassword, $cipherTextPassword);

        $passwordEncryptResult = NULL;
        echo "Password: " . $passwords[$index] . "<br>";
        echo "Hasil Enkripsi Password: ";
        for ($i = 0; $i < 16; ++$i) {
            $passwordEncryptResult .= sprintf('%02x', $cipherTextPassword[$i]);
        }
        echo $passwordEncryptResult;
        echo "<br>";

        if(!($dataLoginUsername == "admin" && $dataLoginPassword == "admin")){
            // Enkripsi role_pekerjaan menggunakan fungsi encryptAES dengan key yang sama
            $keyDataRole = FFI::new("unsigned char[" . strlen($key) . "]");
            FFI::memcpy($keyDataRole, $key, strlen($key));
            $plainTextCDataRole = FFI::new("unsigned char[" . strlen($roles[$index]) . "]");
            FFI::memcpy($plainTextCDataRole, $roles[$index], strlen($roles[$index]));
        
            $cipherTextRole = FFI::new("unsigned char[16]");
        
            $encrypt->encryptAES($plainTextCDataRole, $keyDataRole, $cipherTextRole);

            $roleEncryptResult = NULL;
            echo "Role Pekerjaan: " . $roles[$index] . "<br>";
            echo "Hasil Enkripsi Role Pekerjaan: ";
            for ($i = 0; $i < 16; ++$i) {
                $roleEncryptResult .= sprintf('%02x', $cipherTextRole[$i]);
            }
            echo $roleEncryptResult;

            echo "<br>";
            $index = $index + 1;
            $sql = "UPDATE user_encrypted SET username ='$usernameEncryptResult', password ='$passwordEncryptResult', role_pekerjaan = '$roleEncryptResult'  WHERE id = $index";
            $conn->query($sql);
            break;
        }
        echo "<br>";
        $index = $index + 1;
        $sql = "UPDATE admin_encrypted SET username ='$usernameEncryptResult', password ='$passwordEncryptResult' WHERE id = $index";
        $conn->query($sql);
        break;
    }
}

// Menutup koneksi
mysqli_close($conn);

?>